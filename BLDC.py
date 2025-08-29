# bldc_hall_sim_nocli.py
# Self-contained BLDC 120° Hall simulator with interactive-first plotting.
# Edit CONFIG below and run:  python bldc_hall_sim_nocli.py

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
    "hall_sequence": (5, 1, 3, 2, 6, 4),  # decimal of (HA<<2)|(HB<<1)|HC, forward order

    # Speed profile: "ramp" | "step" | "trapezoid"
    "profile_kind": "ramp",
    "rpm_start": 0.0,
    "rpm_end": 780.0,
    "t_step": 0.3,                # step time (for step/trapezoid)
    "t_flat": 0.6,                # flat duration (for trapezoid)

    # Noise/robustness
    "jitter_deg_electrical": 0.0, # random electrical-angle dither (deg)
    "glitch_prob": 0.0,           # per-sample bit flip probability
    "rng_seed": 2025,

    # Back-EMF (default sinusoidal)
    "bemf_type": "sin",           # "sin" | "trap"
    "ke": 0.297,                   # V·s/rad if proportional; else volts
    "bemf_prop_speed": True,      # True -> E = Ke*omega_mech*shape

    # Outputs
    "write_csv": True,
    "export_pdf": False,          # also export all plots into hall_sim_report.pdf

    # Show figures first, save after closing
    "interactive_first": True,
    "mpl_backend": "TkAgg",       # Try "TkAgg" | "Qt5Agg" | "MacOSX"

    # Optional overlay runs on a separate figure:
    # Each tuple = (label, profile_kind, rpm_start, rpm_end, t_step, t_flat)
    "rpm_overlay_runs": [
        # ("Ramp100-400", "ramp", 100, 400, 0.3, 0.6),
        # ("Step0-780@0.5s", "step", 0, 780, 0.5, 0.6),
        # ("Trap100-400", "trapezoid", 100, 400, 0.3, 0.8),
    ],
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

def compute_bemf(theta_e_rad, omega_mech, bemf_type="sin", ke=0.02, proportional=True):
    """
    Returns (ea, eb, ec).
    - theta_e_rad: electrical angle (rad)
    - omega_mech : mechanical angular speed (rad/s)
    - bemf_type  : 'sin' or 'trap' (120°)
    - ke         : if proportional=True, volts·s/rad (E = ke * omega_mech * shape)
                   else constant volts (E = ke * shape)
    """
    theta_deg = np.degrees(theta_e_rad) % 360.0
    thA = theta_deg
    thB = (theta_deg - 120.0) % 360.0
    thC = (theta_deg - 240.0) % 360.0

    if bemf_type == "sin":
        shapeA = np.sin(np.radians(thA))
        shapeB = np.sin(np.radians(thB))
        shapeC = np.sin(np.radians(thC))
    else:
        shapeA = trap120_unit(thA)
        shapeB = trap120_unit(thB)
        shapeC = trap120_unit(thC)

    amp = (ke * omega_mech) if proportional else ke
    if np.isscalar(amp): amp = np.full_like(shapeA, amp, dtype=float)
    return amp*shapeA, amp*shapeB, amp*shapeC

def overlay_rpm_runs(duration_s, fs, runs):
    if not runs:
        return None
    t = np.arange(0.0, duration_s, 1.0/fs)
    fig = plt.figure(figsize=(11,4))
    for (label, profile_kind, rpm_start, rpm_end, t_step, t_flat) in runs:
        rpm = rpm_profile(t, kind=profile_kind, rpm_start=rpm_start,
                          rpm_end=rpm_end, t_step=t_step, t_flat=t_flat)
        plt.plot(t, rpm, label=f"{label} ({profile_kind} {rpm_start}→{rpm_end})")
    plt.xlabel("Time (s)"); plt.ylabel("RPM")
    plt.title("RPM vs Time — Overlaid Runs")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    return fig

# ---------------- Core sim ----------------

def simulate_halls(cfg):
    rng = np.random.default_rng(cfg["rng_seed"])
    t = np.arange(0.0, cfg["duration_s"], 1.0/cfg["fs"])
    rpm = rpm_profile(t, cfg["profile_kind"], cfg["rpm_start"], cfg["rpm_end"],
                      cfg["t_step"], cfg["t_flat"])

    # Mechanical & electrical angles
    omega_mech = rpm * 2*np.pi / 60.0
    theta_mech = np.cumsum(omega_mech) * (1.0/cfg["fs"])
    theta_e = cfg["pole_pairs"] * theta_mech
    if cfg["jitter_deg_electrical"] > 0.0:
        jitter_rad = np.deg2rad(cfg["jitter_deg_electrical"])
        theta_e = theta_e + rng.uniform(-jitter_rad, +jitter_rad, size=theta_e.size)

    # Halls
    sector = ((theta_e % (2*np.pi)) // (np.pi/3)).astype(int)
    valid_states = np.array(cfg["hall_sequence"], dtype=int)
    state = valid_states[sector]

    HA = ((state >> 2) & 1).astype(np.uint8)
    HB = ((state >> 1) & 1).astype(np.uint8)
    HC = (state & 1).astype(np.uint8)

    # Optional glitches
    if cfg["glitch_prob"] > 0.0:
        flipsA = rng.random(HA.size) < cfg["glitch_prob"]
        flipsB = rng.random(HB.size) < cfg["glitch_prob"]
        flipsC = rng.random(HC.size) < cfg["glitch_prob"]
        HA[flipsA] ^= 1; HB[flipsB] ^= 1; HC[flipsC] ^= 1
        state = (HA.astype(int) << 2) | (HB.astype(int) << 1) | HC.astype(int)

    # Hall-based staircase angle (hold-last for invalids)
    idx_map = {s:i for i,s in enumerate(valid_states)}
    hall_idx = np.empty_like(state, dtype=int)
    last = 0
    for i, s in enumerate(state):
        if s in idx_map: last = idx_map[s]
        hall_idx[i] = last

    theta_e_deg_true = (np.degrees(theta_e) % 360.0)
    theta_e_deg_hall = hall_idx * 60.0

    # Default EMF (we regenerate in plotting with CONFIG)
    ea, eb, ec = compute_bemf(theta_e, omega_mech, bemf_type="sin", ke=0.02, proportional=True)

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

def plot_all(sim, cfg):
    figs = []

    # 1) Hall waveforms — clearer lanes (HA+4, HB+2, HC+0)
    fig1 = plt.figure(figsize=(11,4))
    t, HA, HB, HC = sim["t"], sim["HA"], sim["HB"], sim["HC"]
    plt.step(t, HA*1.0 + 4, where='post', label='HA', color="red")
    plt.step(t, HB*1.0 + 2, where='post', label='HB', color="green")
    plt.step(t, HC*1.0 + 0, where='post', label='HC', color="blue")
    plt.yticks([0,2,4], ['HC','HB','HA'])
    plt.xlabel('Time (s)')
    seq = "→".join(str(s) for s in sim["hall_sequence"])
    plt.title(f'Hall Sensor Waveforms (Seq {seq})')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    figs.append(fig1)

    # 2) True electrical angle
    fig2 = plt.figure(figsize=(11,4))
    plt.plot(sim["t"], sim["theta_e_deg_true"], label='Electrical angle (true, 0..360)', color="blue")
    plt.xlabel('Time (s)'); plt.ylabel('Angle (deg)')
    plt.title('True Electrical Angle')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    figs.append(fig2)

    # 3) Hall-based angle estimate
    fig3 = plt.figure(figsize=(11,4))
    stair = sim["theta_e_deg_hall"]; state = sim["HallState"]
    plt.step(sim["t"], stair, where='post', label='Hall-based angle estimate (staircase)', color="orange")
    jump_idx = np.where(np.diff(stair) != 0)[0] + 1
    for i in jump_idx[:12]:
        plt.text(sim["t"][i], stair[i] + 3, f"{state[i]}", fontsize=9, ha='center', va='bottom')
    plt.xlabel('Time (s)'); plt.ylabel('Angle (deg)')
    plt.title('Hall-Based Electrical Angle Estimate (early states annotated)')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    figs.append(fig3)

    # 4) RPM vs time
    fig4 = plt.figure(figsize=(11,4))
    plt.plot(sim["t"], sim["rpm"], label="Mechanical RPM", color="purple")
    plt.xlabel('Time (s)'); plt.ylabel('RPM')
    plt.title('Mechanical RPM vs Time')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    figs.append(fig4)

    # 5) Electrical vs Mechanical frequency map
    fig5, ax1 = plt.subplots(figsize=(9,5))
    f_mech_map = np.linspace(0, 100, 500)                         # Hz (0..100 Hz -> 0..6000 RPM)
    f_elec_map = sim["pole_pairs"] * f_mech_map                   # Hz
    ax1.plot(f_mech_map, f_elec_map, color="blue", label=f'f_elec = {sim["pole_pairs"]} × f_mech')
    ax1.set_xlabel('Mechanical frequency f_mech (Hz)')
    ax1.set_ylabel('Electrical frequency f_elec (Hz)')
    ax1.set_title(f'Electrical vs Mechanical Frequency (pole_pairs = {sim["pole_pairs"]})')
    ax1.grid(True, alpha=0.3); ax1.legend(loc="upper left")
    def hz_to_rpm(x): return x * 60
    def rpm_to_hz(x): return x / 60
    ax2 = ax1.secondary_xaxis('top', functions=(hz_to_rpm, rpm_to_hz))
    ax2.set_xlabel('Mechanical speed (RPM)')
    def hz_to_rads(x): return x * 2*np.pi
    def rads_to_hz(x): return x / (2*np.pi)
    ax3 = ax1.secondary_yaxis('right', functions=(hz_to_rads, rads_to_hz))
    ax3.set_ylabel('Electrical angular speed ωₑ (rad/s)')
    ax1.text(0.02, 0.98,
             '\n'.join((r'$f_m=\mathrm{RPM}/60$', r'$f_e=p f_m$', r'$\omega_e=2\pi f_e$', fr'$p={sim["pole_pairs"]}$')),
             transform=ax1.transAxes, fontsize=11, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    fig5.tight_layout()
    figs.append(fig5)

    # 6) 3-phase Back-EMF
    fig6 = plt.figure(figsize=(11,4))
    ea, eb, ec = compute_bemf(sim["theta_e_deg_true"]*np.pi/180.0,
                              sim["omega_mech"], cfg["bemf_type"], cfg["ke"], cfg["bemf_prop_speed"])
    plt.plot(sim["t"], ea, label="e_a", color="red")
    plt.plot(sim["t"], eb, label="e_b", color="green")
    plt.plot(sim["t"], ec, label="e_c", color="blue")
    shape = "Sinusoidal" if cfg["bemf_type"]=="sin" else "Trapezoidal (120°)"
    scale = "∝ speed" if cfg["bemf_prop_speed"] else "const amplitude"
    plt.xlabel('Time (s)'); plt.ylabel('Back-EMF (V)')
    plt.title(f'3-Phase Back-EMF — {shape}, {scale} (Ke={cfg["ke"]})')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    figs.append(fig6)

    # 7) RPM + mechanical frequency + electrical frequency vs time
    fig7, ax1 = plt.subplots(figsize=(11,4))
    t = sim["t"]; rpm = sim["rpm"]
    f_mech = rpm/60.0; f_elec = sim["pole_pairs"]*f_mech
    ax1.plot(t, rpm, color="purple", label="Mechanical RPM")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("RPM", color="purple")
    ax1.tick_params(axis='y', labelcolor="purple")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(t, f_mech, color="blue", label="f_mech (Hz)")
    ax2.plot(t, f_elec, color="red", linestyle="--", label="f_elec (Hz)")
    ax2.set_ylabel("Frequency (Hz)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines+lines2, labels+labels2, loc="upper left")
    plt.title("RPM, Mechanical Frequency (Hz), and Electrical Frequency (Hz) vs Time")
    plt.tight_layout()
    figs.append(fig7)

    # Optional overlay figure
    ov = overlay_rpm_runs(cfg["duration_s"], cfg["fs"], cfg["rpm_overlay_runs"])
    if ov is not None:
        figs.append(ov)

    return figs

# ---------- Saving ----------

def save_all(figs, cfg):
    # Save PNGs in the same order
    names = [
        "plot_1_hall_waveforms.png",
        "plot_2_true_elec_angle.png",
        "plot_3_hall_angle_est.png",
        "plot_4_rpm_vs_time.png",
        "plot_5_freq_map.png",
        "plot_6_bemf.png",
        "plot_7_rpm_freq.png",
        "overlay_rpm.png",          # only if overlay was created
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

    # Build figures (no saving yet)
    figs = plot_all(sim, cfg)

    # POP-UP WINDOWS NOW; blocks until you close them
    plt.show()

    # After windows are closed: save files
    save_all(figs, cfg)
    if cfg["write_csv"]:
        dump_csv(sim, "hall_sim_output.csv")
    print("Done. PNGs and CSV saved to current folder.")

if __name__ == "__main__":
    main()

