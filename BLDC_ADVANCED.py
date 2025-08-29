# bldc_hall_sim_nocli.py
# BLDC 120° Hall simulator — grouped windows, Hall frequency annotations, interactive-first.

import numpy as np
import pandas as pd
import matplotlib as mpl

# ============================== CONFIG ==============================
CONFIG = {
    # Simulation
    "duration_s": 0.2,
    "fs": 10_000,                 # Hz

    # Motor & Halls
    "pole_pairs": 15,
    "hall_sequence": (5, 1, 3, 2, 6, 4),  # decimal of (HC<<2)|(HB<<1)|HA, forward order

    # Speed profile: "ramp" | "step" | "trapezoid"
    "profile_kind": "ramp",
    "rpm_start": 780.0,
    "rpm_end": 0.0,
    "t_step": 0.3,                # step time (for step/trapezoid)
    "t_flat": 0.6,                # flat duration (for trapezoid)

    # Noise/robustness
    "jitter_deg_electrical": 0.0, # random electrical-angle dither (deg)
    "glitch_prob": 0.0,           # per-sample bit flip probability
    "rng_seed": 2025,

    # Back-EMF (sinusoidal by default)
    "bemf_type": "sin",           # "sin" | "trap"
    "ke": 0.297,                  # V·s/rad (per-phase peak per mechanical rad/s)
    "bemf_prop_speed": True,      # True -> E = Ke*omega_mech*shape

    # Outputs
    "write_csv": True,
    "export_pdf": False,          # also export all plots into hall_sim_report.pdf

    # Show figures first, save after closing
    "interactive_first": True,
    "mpl_backend": "TkAgg",       # Try "TkAgg" | "Qt5Agg" | "MacOSX"

    # RPM overlay runs (inset inside Figure C)
    # Each tuple = (label, profile_kind, rpm_start, rpm_end, t_step, t_flat)
    "rpm_overlay_runs": [
        # ("Ramp100-400", "ramp", 100, 400, 0.3, 0.6),
        # ("Step0-780@0.1s", "step", 0, 780, 0.1, 0.6),
        # ("Trap100-700", "trapezoid", 100, 700, 0.2, 0.4),
    ],

    # Annotate Hall sensor frequency at these speeds (RPM) on Hall waveforms (Fig A) and RPM plot (Fig C)
    # f_hall = f_elec = pole_pairs * (RPM/60)  [Hz]
    "hall_freq_markers_rpm": ["end", 300.0, 600.0],  # "end" -> rpm_end
}
# ====================================================================

# Set GUI backend for pop-up windows (before importing pyplot)
if CONFIG.get("mpl_backend"):
    try:
        mpl.use(CONFIG["mpl_backend"])
    except Exception as e:
        print(f"[Warn] Could not set backend {CONFIG['mpl_backend']}: {e}")

import matplotlib.pyplot as plt

# ---------------- Helpers ----------------

def rpm_profile(t, kind="ramp", rpm_start=100.0, rpm_end=400.0, t_step=0.3, t_flat=0.6):
    T = t[-1] if t.size else 0.0
    rpm = np.empty_like(t)
    if kind == "ramp":
        rpm[:] = np.linspace(rpm_start, rpm_end, t.size)
    elif kind == "step":
        rpm[:] = rpm_start
        rpm[t >= t_step] = rpm_end
    elif kind == "trapezoid":
        t1, t2, t3 = t_step, t_step + t_flat, T
        rpm[:] = rpm_start
        m_up = (rpm_end - rpm_start) / max(t1, 1e-9)
        mask_up = (t >= 0) & (t < t1)
        rpm[mask_up] = rpm_start + m_up * (t[mask_up] - 0.0)
        mask_flat = (t >= t1) & (t < t2)
        rpm[mask_flat] = rpm_end
        m_dn = (rpm_start - rpm_end) / max((t3 - t2), 1e-9)
        mask_dn = (t >= t2)
        rpm[mask_dn] = rpm_end + m_dn * (t[mask_dn] - t2)
    else:
        raise ValueError("Unknown profile kind")
    return rpm

def _wrap_deg(x): return (x + 180.0) % 360.0 - 180.0

def trap120_unit(theta_deg):
    a = _wrap_deg(theta_deg)
    out = np.empty_like(a, dtype=float)
    m1 = (a >= -150) & (a < -30)
    m2 = (a >= -30)  & (a < 30)
    m3 = (a >= 30)   & (a < 150)
    m4 = (a >= 150)  & (a <= 180)
    m5 = (a >= -180) & (a < -150)
    out[m1] = -1.0
    out[m2] = a[m2] / 30.0           # -1 → +1 across 60°
    out[m3] = +1.0
    out[m4] = (180.0 - a[m4]) / 30.0 # +1 → 0
    out[m5] = (a[m5] + 180.0) / 30.0 # 0 → -1
    return out


# Compute back-EMF voltages (EA, EB, EC) for a given electrical angle (rad) and speed (rad/s)
# Returns (ea, eb, ec).
# - theta_e_rad: electrical angle (rad)
# - omega_mech : mechanical angular speed (rad/s)
# - bemf_type  : 'sin' or 'trap' (120°); sin is referred as sinusoidal while trap is trapezoidal
# - ke         : if proportional=True, volts·s/rad (E = ke * omega_mech * shape)
#                else constant volts (E = ke * shape)
def compute_bemf(theta_e_rad, omega_mech, bemf_type="sin", ke=0.02, proportional=True):
    """
    Returns (ea, eb, ec).
    - theta_e_rad: electrical angle (rad)
    - omega_mech : mechanical angular speed (rad/s)
    - bemf_type  : 'sin' or 'trap' (120°)
    - ke         : if proportional=True, volts·s/rad (E = ke * omega_mech * shape)
                   else constant volts (E = ke * shape)
    """
    # Compute electrical angles for each phase
    theta_deg = np.degrees(theta_e_rad) % 360.0
    # Compute phase angles
    # 120° apart for 3-phase system
    # The purpose of wrapping the angle is to keep it within the range [0, 360) degrees
    # Numerical Example of Wrapping: 370° -> 10°
    # The formula is: (x + 180) % 360 - 180
    thA = theta_deg    # Phase A
    thB = (theta_deg - 120.0) % 360.0 # Phase B, the purpose of this is to create a 120° phase shift, while %360 is used to wrap the angle
    thC = (theta_deg - 240.0) % 360.0 # Phase C, the purpose of this is to create a 240° phase shift, while %360 is used to wrap the angle

    # Compute BEMF shapes
    # Sinusoidal BEMF
    # Shape is a sine wave
    # The formula is: E = ke * omega_mech * sin(theta_e)
    if bemf_type == "sin":
        shapeA = np.sin(np.radians(thA))
        shapeB = np.sin(np.radians(thB))
        shapeC = np.sin(np.radians(thC))
    else:
        # Trapezoidal BEMF
        # Shape is a trapezoidal wave
        shapeA = trap120_unit(thA)
        shapeB = trap120_unit(thB)
        shapeC = trap120_unit(thC)

    #The following statements is to compute the amplitude
    # The amplitude is determined based on whether it is proportional to speed or constant
    # If proportional, the amplitude scales with speed; if constant, it is fixed
    amp = (ke * omega_mech) if proportional else ke
    # If the amplitude is a scalar, expand it to match the shape of the BEMF waveforms
    # This ensures that the amplitude is applied correctly to each phase
    # For example, if amp is a scalar and shapeA is an array, we create a new array filled with amp
    # This is done using np.full_like to create an array of the same shape as shapeA
    # amp is scalar when it is a single value (not an array), so we need to expand it
    if np.isscalar(amp): amp = np.full_like(shapeA, amp, dtype=float)
    return amp*shapeA, amp*shapeB, amp*shapeC

# ---------------- Core sim ----------------
# Simulate Hall sensor signals
# Returns a dictionary with the simulated signals
# Input Argument: cfg (configuration dictionary)
def simulate_halls(cfg):
    # Initialize random number generator
    # Set seed for reproducibility
    rng = np.random.default_rng(cfg["rng_seed"])

    # Generate time vector
    # The time vector is generated based on the duration and sampling frequency
    # It defines the discrete time steps for the simulation
    # The total number of samples is determined by the duration and sampling frequency
    t = np.arange(0.0, cfg["duration_s"], 1.0/cfg["fs"])

    # Generate RPM profile
    # The RPM profile defines how the motor speed changes over time
    # It is typically a function of time and can have different shapes (e.g., linear, step)
    # In this case, we use a user-defined function to generate the RPM values
    rpm = rpm_profile(t, cfg["profile_kind"], cfg["rpm_start"], cfg["rpm_end"],
                      cfg["t_step"], cfg["t_flat"])

    # Mechanical & electrical angles
    # Compute mechanical and electrical angles
    # Mechanical angular velocity (rad/s)
    # omega_mech = rpm * 2*np.pi / 60.0
    omega_mech = rpm * 2*np.pi / 60.0

    # Mechanical angle (rad)
    # The definition of mechanical angle is the angle of the rotor in the mechanical frame
    # Simply speaking, it is the angle that the rotor has rotated from a reference position
    # So Mechanical Angle which is also called angular position is the integral of angular velocity over time
    # theta_mech = np.cumsum(omega_mech) * (1.0/cfg["fs"])
    # np.cumsum is to integrate the angular velocity to get the mechanical angle
    # as the formula of mechanical angle is: θ(t) = ∫₀ᵗ ω(τ) dτ 
    # for discrete approximation:
    # θ[n] = Σᵢ₌₀ⁿ ω[i] × Δt
    theta_mech = np.cumsum(omega_mech) * (1.0/cfg["fs"])

    # Electrical angle (rad)
    # The electrical angle is related to the mechanical angle by the number of pole pairs
    # Specifically, it is given by the formula: θ_e = p * θ_m where p is the number of pole pairs and θ_m is the mechanical angle
    theta_e = cfg["pole_pairs"] * theta_mech

    # Add jitter to electrical angle
    # This simulates variations in the electrical angle due to sensor noise or other factors
    # The jitter is applied uniformly within a specified range
    # For example, if jitter_deg_electrical is 5 degrees, the electrical angle will be randomly perturbed by ±5 degrees
    # This is done to simulate real-world sensor noise
    if cfg["jitter_deg_electrical"] > 0.0:
        # Convert jitter from degrees to radians
        jitter_rad = np.deg2rad(cfg["jitter_deg_electrical"])
        # Apply jitter to electrical angle
        theta_e = theta_e + rng.uniform(-jitter_rad, +jitter_rad, size=theta_e.size)

    # Halls Sensor
    # Simulate Hall sensor signals based on electrical angle
    # The Hall sensor signals are typically 120 degrees out of phase with each other
    # This means that when one sensor is at its maximum, the others are at their minimum
    # The sector is determined by the electrical angle
    # Specifically, it is given by the formula: sector = (theta_e % (2*np.pi)) // (np.pi/3)
    # There are totally 6 sectors (0 to 5) in a full electrical rotation
    # Each sector corresponds to a specific Hall sensor state
    # The mapping is defined in the hall_sequence configuration
    sector = ((theta_e % (2*np.pi)) // (np.pi/3)).astype(int)
    # There are totally 6 sectors (0 to 5) in a full electrical rotation
    # Each sector corresponds to a specific Hall sensor state
    # The mapping is defined in the hall_sequence configuration
    # valid_states = np.array(cfg["hall_sequence"], dtype=int)
    valid_states = np.array(cfg["hall_sequence"], dtype=int)
    state = valid_states[sector]

    HA = (state & 1).astype(np.uint8)
    HB = ((state >> 1) & 1).astype(np.uint8)
    HC = ((state >> 2) & 1).astype(np.uint8)

    # Optional glitches
    if cfg["glitch_prob"] > 0.0:
        flipsA = rng.random(HA.size) < cfg["glitch_prob"]
        flipsB = rng.random(HB.size) < cfg["glitch_prob"]
        flipsC = rng.random(HC.size) < cfg["glitch_prob"]
        HA[flipsA] ^= 1; HB[flipsB] ^= 1; HC[flipsC] ^= 1
        state = (HC.astype(int) << 2) | (HB.astype(int) << 1) | HA.astype(int)

    # Hall-based staircase angle (hold-last for invalids)
    idx_map = {s:i for i,s in enumerate(valid_states)}
    hall_idx = np.empty_like(state, dtype=int)
    last = 0
    for i, s in enumerate(state):
        if s in idx_map: last = idx_map[s]
        hall_idx[i] = last

    theta_e_deg_true = (np.degrees(theta_e) % 360.0)
    theta_e_deg_hall = hall_idx * 60.0

    ea, eb, ec = compute_bemf(theta_e, omega_mech, bemf_type=cfg["bemf_type"],
                              ke=cfg["ke"], proportional=cfg["bemf_prop_speed"])

    return {
        "t": t, "rpm": rpm,
        "theta_mech_rad": theta_mech,
        "theta_e_rad": theta_e,
        "theta_e_deg_true": theta_e_deg_true,
        "theta_e_deg_hall": theta_e_deg_hall,
        "omega_mech": omega_mech,
        "HA": HA, "HB": HB, "HC": HC,
        "HallState": state.astype(int),
        "pole_pairs": cfg["pole_pairs"],
        "hall_sequence": valid_states,
        "ea": ea, "eb": eb, "ec": ec,
    }

def dump_csv(sim, path="hall_sim_output.csv"):
    df = pd.DataFrame({
        "t_s": sim["t"], "RPM": sim["rpm"],
        "theta_mech_rad": sim["theta_mech_rad"],
        "theta_e_deg_true": sim["theta_e_deg_true"],
        "theta_e_deg_hall_est": sim["theta_e_deg_hall"],
        "HA": sim["HA"], "HB": sim["HB"], "HC": sim["HC"],
        "HallState": sim["HallState"],
    })
    df.to_csv(path, index=False)
    print(f"Saved CSV -> {path}")

# ---------- Plot builders (return figures; do not save here) ----------

def _resolve_markers():
    """Return list of RPM values to annotate."""
    markers = CONFIG.get("hall_freq_markers_rpm", [])
    vals = []
    for m in markers:
        if isinstance(m, str) and m.lower() == "end":
            vals.append(float(CONFIG["rpm_end"]))
        else:
            try:
                vals.append(float(m))
            except:
                pass
    return vals

def plot_all(sim, cfg):
    figs = []
    t = sim["t"]
    markers_rpm = _resolve_markers()

    # ===================== FIGURE A: HALL BLOCK (3 subplots) =====================
    figA, axsA = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # A1) Hall waveforms — clearer lanes
    ax = axsA[0]
    ax.step(t, sim["HA"]*1.0 + 4, where='post', label='HA', color="red")
    ax.step(t, sim["HB"]*1.0 + 2, where='post', label='HB', color="green")
    ax.step(t, sim["HC"]*1.0 + 0, where='post', label='HC', color="blue")
    ax.set_yticks([0,2,4], ['HC','HB','HA'])
    ax.set_ylabel("Hall")
    seq = "→".join(str(s) for s in sim["hall_sequence"])
    ax.set_title(f'Hall Sensor Waveforms (Seq {seq})')
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right")

    # Hall frequency annotations on Hall waveforms
    for rpm_mark in markers_rpm:
        idx = int(np.argmin(np.abs(sim["rpm"] - rpm_mark)))
        t_mark = t[idx]; rpm_val = sim["rpm"][idx]
        f_mech = rpm_val / 60.0
        f_hall = CONFIG["pole_pairs"] * f_mech
        ax.axvline(t_mark, color="gray", linestyle=":", alpha=0.6)
        ax.text(t_mark, 5.0,
                f"RPM≈{rpm_val:.0f}\nHall f≈{f_hall:.1f} Hz",
                fontsize=8, va="bottom", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    # A2) True electrical angle
    ax = axsA[1]
    ax.plot(t, sim["theta_e_deg_true"], color="blue", label="Electrical angle (true)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("True Electrical Angle")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right")

    # A3) Hall-based angle estimate
    ax = axsA[2]
    stair = sim["theta_e_deg_hall"]; state = sim["HallState"]
    ax.step(t, stair, where='post', label="Hall-based estimate", color="orange")
    jump_idx = np.where(np.diff(stair) != 0)[0] + 1
    for i in jump_idx[:12]:
        ax.text(t[i], stair[i] + 3, f"{state[i]}", fontsize=9,
                ha='center', va='bottom')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (deg)")
    ax.set_title("Hall-Based Electrical Angle Estimate")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right")

    figA.tight_layout()
    figs.append(figA)

    # ================= FIGURE B: BACK-EMF (standalone) =================
    figB, axB = plt.subplots(1, 1, figsize=(12, 4))
    axB.plot(t, sim["ea"], label="e_a", color="red")
    axB.plot(t, sim["eb"], label="e_b", color="green")
    axB.plot(t, sim["ec"], label="e_c", color="blue")
    shape = "Sinusoidal" if CONFIG["bemf_type"]=="sin" else "Trapezoidal (120°)"
    scale = "∝ speed" if CONFIG["bemf_prop_speed"] else "const amplitude"
    axB.set_xlabel("Time (s)")
    axB.set_ylabel("Back-EMF (V)")
    axB.set_title(f"3-Phase Back-EMF — {shape}, {scale} (Ke={CONFIG['ke']})")
    axB.grid(True, alpha=0.3); axB.legend(loc="upper right")
    figB.tight_layout()
    figs.append(figB)

    # ================= FIGURE C: RPM vs TIME + FREQ MAP (2 subplots) =================
    figC, (ax_rpm, ax_map) = plt.subplots(1, 2, figsize=(14, 5))

    # C1) RPM vs time
    ax_rpm.plot(t, sim["rpm"], label="Mechanical RPM", color="purple")
    ax_rpm.set_title("Mechanical RPM vs Time")
    ax_rpm.set_xlabel("Time (s)")
    ax_rpm.set_ylabel("RPM")
    ax_rpm.grid(True, alpha=0.3); ax_rpm.legend(loc="upper left")

    # Hall frequency annotations on RPM plot
    for rpm_mark in markers_rpm:
        idx = int(np.argmin(np.abs(sim["rpm"] - rpm_mark)))
        t_mark = t[idx]; rpm_val = sim["rpm"][idx]
        f_mech = rpm_val / 60.0
        f_hall = CONFIG["pole_pairs"] * f_mech
        ax_rpm.axvline(t_mark, color="gray", linestyle=":", alpha=0.6)
        ax_rpm.plot(t_mark, rpm_val, 'o', color="black", ms=4)
        ax_rpm.text(t_mark, rpm_val,
                    f"\nRPM≈{rpm_val:.0f}\nHall f≈{f_hall:.1f} Hz",
                    fontsize=9, va="bottom", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    # Inset for overlay runs
    runs = CONFIG.get("rpm_overlay_runs", [])
    if runs:
        inset = ax_rpm.inset_axes([0.55, 0.15, 0.4, 0.6])  # [left, bottom, w, h]
        inset.grid(True, alpha=0.3)
        inset.set_title("Overlays", fontsize=10)
        for (label, profile_kind, rpm_start, rpm_end, t_step, t_flat) in runs:
            rpm_ov = rpm_profile(t, kind=profile_kind,
                                 rpm_start=rpm_start, rpm_end=rpm_end,
                                 t_step=t_step, t_flat=t_flat)
            inset.plot(t, rpm_ov, label=label)
        inset.set_xlabel("t (s)", fontsize=9)
        inset.set_ylabel("RPM", fontsize=9)
        inset.tick_params(labelsize=8)
        inset.legend(fontsize=8, loc="upper left", ncol=1, framealpha=0.8)

    # C2) Electrical vs Mechanical frequency map
    f_mech_map = np.linspace(0, 100, 500)  # Hz
    f_elec_map = CONFIG["pole_pairs"] * f_mech_map
    ax_map.plot(f_mech_map, f_elec_map, color="blue",
                label=f"f_elec = {CONFIG['pole_pairs']} × f_mech")
    ax_map.set_xlabel("f_mech (Hz)"); ax_map.set_ylabel("f_elec (Hz)")
    ax_map.set_title(f"Electrical vs Mechanical Frequency (p = {CONFIG['pole_pairs']})")
    ax_map.grid(True, alpha=0.3); ax_map.legend(loc="upper left")
    # secondary axes
    def hz_to_rpm(x): return x * 60
    def rpm_to_hz(x): return x / 60
    ax_map.secondary_xaxis("top", functions=(hz_to_rpm, rpm_to_hz)).set_xlabel("Mechanical speed (RPM)")
    def hz_to_rads(x): return x * 2*np.pi
    def rads_to_hz(x): return x / (2*np.pi)
    ax_map.secondary_yaxis("right", functions=(hz_to_rads, rads_to_hz)).set_ylabel("ωe (rad/s)")

    figC.tight_layout()
    figs.append(figC)

    return figs

# ---------- Saving ----------

def save_all(figs, cfg):
    # Save PNGs in window order
    names = [
        "window_A_hall_block.png",
        "window_B_back_emf.png",
        "window_C_rpm_and_map.png",
    ]
    for i, f in enumerate(figs):
        fname = names[i] if i < len(names) else f"plot_extra_{i+1}.png"
        try:
            f.savefig(fname, dpi=150)
        except Exception as e:
            print(f"[Warn] Could not save {fname}: {e}")
    if cfg["export_pdf"]:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages("hall_sim_report.pdf") as pdf:
            for f in figs:
                pdf.savefig(f, bbox_inches='tight')
        print("Saved PDF -> hall_sim_report.pdf")

# ---------- Main ----------

def main():
    cfg = CONFIG
    sim = simulate_halls(cfg)

    # Build grouped-figure windows (no saving yet)
    figs = plot_all(sim, cfg)

    # POP-UP WINDOWS NOW; blocks until you close them
    plt.show()

    # After windows are closed: save files and CSV
    save_all(figs, cfg)
    if cfg["write_csv"]:
        dump_csv(sim, "hall_sim_output.csv")
    print("Done. PNGs and CSV saved to current folder.")

if __name__ == "__main__":
    main()
