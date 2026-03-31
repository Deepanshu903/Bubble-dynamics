import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
import tempfile
import os
import io
import csv

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bubble Column Analyser",
    page_icon="🫧",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}
.stApp {
    background: #0d1117;
    color: #e6edf3;
}
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    padding: 10px 24px;
    width: 100%;
    transition: all 0.2s ease;
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950);
    box-shadow: 0 0 12px rgba(46,160,67,0.4);
    transform: translateY(-1px);
}
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-card .label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: #58a6ff;
}
.metric-card .unit {
    font-size: 12px;
    color: #8b949e;
    margin-top: 2px;
}
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #8b949e;
    border-bottom: 1px solid #30363d;
    padding-bottom: 8px;
    margin-bottom: 16px;
}
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #238636, #58a6ff);
    border-radius: 4px;
}
.stSlider > div > div > div > div {
    background: #58a6ff !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 24px 0 8px 0;">
    <h1 style="color:#58a6ff; font-size:28px; margin:0;">🫧 Bubble Column Dynamics</h1>
    <p style="color:#8b949e; font-family:'DM Sans',sans-serif; margin-top:6px; font-size:15px;">
        Upload a high-speed video and extract bubble velocity &amp; diameter statistics frame by frame.
    </p>
</div>
<hr style="border-color:#30363d; margin-bottom:24px;">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Sidebar — Inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">📁 Video Input</div>', unsafe_allow_html=True)
    video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

    st.markdown('<div class="section-header" style="margin-top:24px;">⚙️ Calibration</div>', unsafe_allow_html=True)
    mm_per_pixel = st.number_input(
        "mm per Pixel", min_value=0.001, max_value=10.0,
        value=0.125, step=0.001, format="%.4f",
        help="Physical scale: how many mm each pixel represents."
    )
    original_fps = st.number_input(
        "Original FPS", min_value=1, max_value=10000,
        value=240, step=1,
        help="Camera recording frame rate (fps)."
    )

    st.markdown('<div class="section-header" style="margin-top:24px;">✂️ Crop Ratios</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        crop_left   = st.slider("Left",   0.0, 0.5, 0.15, 0.01)
        crop_top    = st.slider("Top",    0.0, 0.5, 0.20, 0.01)
    with col_r:
        crop_right  = st.slider("Right",  0.0, 0.5, 0.30, 0.01)
        crop_bottom = st.slider("Bottom", 0.0, 0.5, 0.35, 0.01)

    st.markdown('<div class="section-header" style="margin-top:24px;">🔍 Detection</div>', unsafe_allow_html=True)
    min_area = st.number_input("Min Bubble Area (px²)", 1, 50000, 100, step=10)
    max_area = st.number_input("Max Bubble Area (px²)", 1, 200000, 10000, step=100)

    st.markdown('<div class="section-header" style="margin-top:24px;">🔗 Matching</div>', unsafe_allow_html=True)
    max_dist = st.slider("Max Match Distance (px)", 5, 200, 60)
    max_dx   = st.slider("Max Horizontal Shift (px)", 1, 100, 25)
    area_tol = st.slider("Area Tolerance", 0.1, 1.0, 0.6, 0.05,
                          help="Fraction of area change allowed between frames.")
    max_vel  = st.slider("Max Velocity (mm/s)", 50, 2000, 400, 10)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Analysis")

# ─────────────────────────────────────────────
#  Helper — smooth
#  reflect-padding avoids zero-pad distortion
#  at the start/end of the signal.
# ─────────────────────────────────────────────
def smooth(x, w=5):
    x = np.array(x, dtype=np.float64)
    if len(x) < w:
        return x
    pad = w // 2
    x_padded = np.pad(x, pad, mode="reflect")
    return np.convolve(x_padded, np.ones(w) / w, mode="valid")[:len(x)]

# ─────────────────────────────────────────────
#  Idle state
# ─────────────────────────────────────────────
if not video_file:
    st.markdown("""
    <div style="text-align:center; padding:80px 20px; color:#8b949e;">
        <div style="font-size:64px; margin-bottom:16px;">🫧</div>
        <p style="font-family:'Space Mono',monospace; font-size:14px;">
            Upload a video in the sidebar to begin analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  Run Analysis
# ─────────────────────────────────────────────
if run_btn:

    # Save upload to a temp file on disk so OpenCV can open it
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.flush()
    tfile.close()

    cap          = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = original_fps if original_fps > 0 else 30

    # Show a live preview snapshot roughly every 5% of the video
    DEBUG_EVERY = max(20, total_frames // 20) if total_frames > 0 else 20

    frame_idx    = 0
    prev_objects = []   # (cx, cy, area) from the previous frame

    # Per-frame time-series lists
    time_axis     = []
    mean_vel      = []
    median_vel    = []
    bubble_counts = []
    mean_diams    = []
    median_diams  = []

    # All individual matched bubble velocities — used for the histogram
    # (more informative than just per-frame means)
    all_bubble_velocities = []

    # ── UI elements ──
    progress_bar = st.progress(0)
    status_txt   = st.empty()

    prev_col1, prev_col2, prev_col3 = st.columns(3)
    preview_slots = [prev_col1.empty(), prev_col2.empty(), prev_col3.empty()]

    # ── Processing loop ──
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Crop
        x1 = int(crop_left         * w)
        x2 = int((1 - crop_right)  * w)
        y1 = int(crop_top          * h)
        y2 = int((1 - crop_bottom) * h)

        # Skip frame if crop ratios are inverted / overlapping
        if x2 <= x1 or y2 <= y1:
            frame_idx += 1
            continue

        frame = frame[y1:y2, x1:x2]

        # Preprocess
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_f    = gray.astype(np.float32)
        mx        = np.max(gray_f)
        gray_norm = (gray_f / mx * 255).astype(np.uint8) if mx > 0 else gray.copy()
        blurred   = cv2.GaussianBlur(gray_norm, (5, 5), 0)
        edges     = cv2.Canny(blurred, 20, 80)
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges     = cv2.dilate(edges, kernel, iterations=1)

        # Contour detection
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        curr_objects = []
        diameters    = []
        vis          = frame.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                curr_objects.append((cx, cy, area))

                # Equivalent circular diameter (px → mm)
                diam = np.sqrt(4.0 * area / np.pi) * mm_per_pixel
                diameters.append(diam)

                xb, yb, wb, hb = cv2.boundingRect(c)
                cv2.rectangle(vis, (xb, yb), (xb + wb, yb + hb), (0, 255, 0), 1)

        # Frame-to-frame matching
        velocities   = []
        used_indices = set()

        for (x_prev, y_prev, area_prev) in prev_objects:
            best_idx  = -1
            best_dist = 1e9

            for idx, (x_curr, y_curr, area_curr) in enumerate(curr_objects):
                if idx in used_indices:
                    continue
                dx   = x_curr - x_prev
                dy   = y_prev - y_curr   # positive = upward
                if dy <= 0:
                    continue
                dist = np.sqrt(dx**2 + dy**2)
                if dist > max_dist:
                    continue
                if abs(dx) > max_dx:
                    continue
                if abs(area_curr - area_prev) / max(area_prev, 1) > area_tol:
                    continue
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = idx

            if best_idx != -1:
                used_indices.add(best_idx)
                _, y_curr, _ = curr_objects[best_idx]
                v_mm = (y_prev - y_curr) * fps * mm_per_pixel
                if 0 < v_mm < max_vel:
                    velocities.append(v_mm)
                    all_bubble_velocities.append(v_mm)  # accumulate for histogram

        # Store per-frame stats
        mean_vel.append(np.mean(velocities)      if velocities else 0.0)
        median_vel.append(np.median(velocities)  if velocities else 0.0)
        mean_diams.append(np.mean(diameters)     if diameters  else 0.0)
        median_diams.append(np.median(diameters) if diameters  else 0.0)
        bubble_counts.append(len(curr_objects))
        time_axis.append(frame_idx / fps)

        prev_objects = curr_objects
        frame_idx   += 1

        # Progress
        pct = frame_idx / total_frames if total_frames > 0 else 0
        progress_bar.progress(min(pct, 1.0))
        status_txt.markdown(
            f"<span style='color:#8b949e; font-family:Space Mono,monospace; font-size:12px;'>"
            f"Frame {frame_idx} / {total_frames} &nbsp;—&nbsp; "
            f"{len(curr_objects)} bubbles detected &nbsp;|&nbsp; "
            f"{len(velocities)} velocity matches</span>",
            unsafe_allow_html=True,
        )

        # Live preview
        if frame_idx % DEBUG_EVERY == 0:
            preview_slots[0].image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                caption="Original", use_container_width=True
            )
            preview_slots[1].image(
                gray_norm, caption="Gray (normalised)",
                clamp=True, use_container_width=True
            )
            preview_slots[2].image(
                cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                caption=f"Detections ({len(curr_objects)})",
                use_container_width=True
            )

    cap.release()
    os.unlink(tfile.name)

    progress_bar.progress(1.0)
    status_txt.markdown(
        "<span style='color:#3fb950; font-family:Space Mono,monospace; font-size:12px;'>"
        "✔ Analysis complete</span>",
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────
    #  Summary metric cards
    # ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Summary</div>', unsafe_allow_html=True)

    avg_count = float(np.mean(bubble_counts)) if bubble_counts else 0.0
    avg_vel   = float(np.mean([v for v in mean_vel   if v > 0])) \
                if any(v > 0 for v in mean_vel)   else 0.0
    avg_diam  = float(np.mean([d for d in mean_diams if d > 0])) \
                if any(d > 0 for d in mean_diams) else 0.0

    m1, m2, m3, m4 = st.columns(4)
    for col, label, value, unit in zip(
        [m1, m2, m3, m4],
        ["Frames Processed", "Avg Bubble Count", "Avg Velocity",  "Avg Diameter"],
        [frame_idx,          f"{avg_count:.1f}", f"{avg_vel:.1f}", f"{avg_diam:.2f}"],
        ["frames",           "bubbles / frame",  "mm/s",           "mm"],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="unit">{unit}</div>
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    #  Plots
    # ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 Results</div>', unsafe_allow_html=True)

    mean_vel_s   = smooth(mean_vel)
    median_vel_s = smooth(median_vel)
    mean_d_s     = smooth(mean_diams)
    median_d_s   = smooth(median_diams)

    DARK_BG  = "#0d1117"
    GRID_COL = "#21262d"
    TEXT_COL = "#8b949e"
    BLUE     = "#58a6ff"
    GREEN    = "#3fb950"
    ORANGE   = "#d29922"

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=TEXT_COL, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.6)
        ax.set_title(title,   color="#e6edf3", fontsize=11, pad=10)
        ax.set_xlabel(xlabel, color=TEXT_COL,  fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT_COL,  fontsize=9)

    fig = plt.figure(figsize=(14, 10), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Plot 1 — Velocity vs Time
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "Bubble Velocity — Mean & Median", "Time (s)", "Velocity (mm/s)")
    ax1.plot(time_axis, mean_vel_s,   color=BLUE,  linewidth=1.5, label="Mean",   alpha=0.9)
    ax1.plot(time_axis, median_vel_s, color=GREEN, linewidth=1.5, label="Median", alpha=0.9)
    ax1.legend(fontsize=8, facecolor="#21262d", labelcolor="#e6edf3", edgecolor=GRID_COL)

    # Plot 2 — Bubble Count vs Time
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "Bubble Count per Frame", "Time (s)", "Count")
    ax2.plot(time_axis, bubble_counts, color=ORANGE, linewidth=1.2, alpha=0.9)

    # Plot 3 — Diameter vs Time
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "Bubble Diameter — Mean & Median", "Time (s)", "Diameter (mm)")
    ax3.plot(time_axis, mean_d_s,   color=BLUE,  linewidth=1.5, label="Mean",   alpha=0.9)
    ax3.plot(time_axis, median_d_s, color=GREEN, linewidth=1.5, label="Median", alpha=0.9)
    ax3.legend(fontsize=8, facecolor="#21262d", labelcolor="#e6edf3", edgecolor=GRID_COL)

    # Plot 4 — Velocity Distribution Histogram (ALL matched bubbles, ALL frames)
    # FIXED: title, x-axis label, and data source all corrected
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4,
             "Velocity Distribution — All Frames",
             "Velocity (mm/s)",           # was "Time (s)" — now correct
             "Frequency (bubble count)")  # clarified y-label
    if all_bubble_velocities:
        ax4.hist(
            all_bubble_velocities,
            bins=40,
            color=BLUE,
            alpha=0.8,
            edgecolor=DARK_BG,
            linewidth=0.4,
        )
        v_mean   = np.mean(all_bubble_velocities)
        v_median = np.median(all_bubble_velocities)
        ax4.axvline(v_mean,   color=GREEN,  linewidth=1.5, linestyle="--",
                    label=f"Mean   {v_mean:.1f} mm/s")
        ax4.axvline(v_median, color=ORANGE, linewidth=1.5, linestyle=":",
                    label=f"Median {v_median:.1f} mm/s")
        ax4.legend(fontsize=8, facecolor="#21262d",
                   labelcolor="#e6edf3", edgecolor=GRID_COL)
    else:
        ax4.text(0.5, 0.5, "No matched bubbles",
                 transform=ax4.transAxes, ha="center", va="center",
                 color=TEXT_COL, fontsize=10)

    st.pyplot(fig)
    plt.close(fig)

    # ─────────────────────────────────────────────
    #  CSV Export
    # ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">💾 Export</div>', unsafe_allow_html=True)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "time_s", "mean_vel_mms", "median_vel_mms",
        "bubble_count", "mean_diam_mm", "median_diam_mm"
    ])
    for row in zip(time_axis, mean_vel, median_vel,
                   bubble_counts, mean_diams, median_diams):
        writer.writerow([f"{v:.4f}" for v in row])

    st.download_button(
        label="⬇  Download CSV",
        data=buf.getvalue(),
        file_name="bubble_column_results.csv",
        mime="text/csv",
    )
