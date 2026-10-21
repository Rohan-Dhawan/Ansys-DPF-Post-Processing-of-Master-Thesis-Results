"""
main.py
-------
Post-processing pipeline for ANSYS DPF / RST simulation files.
Extracts relative tangential micro-slip between bearing ring and shaft nodes,
normalises time axes to shaft rotation angles, and produces publication-quality
comparative plots.

Domain: Wind Turbine Main Bearing Ring Creep
Stack: Python 3.9+, ANSYS DPF (ansys.dpf.post), NumPy, Matplotlib
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Optional import for ANSYS DPF
try:
    from ansys.dpf import post
    HAS_ANSYS_DPF = True
except ImportError:
    HAS_ANSYS_DPF = False


# =============================================================================
# 1. CONFIGURATION & SIMULATION PARAMETERS
# =============================================================================

DEFAULT_CASES = [
    {"path": "data/case_1MNm.rst", "label": "1 MNm", "rot": 4.0},
    {"path": "data/case_2MNm.rst", "label": "2 MNm", "rot": 4.0},
    {"path": "data/case_4MNm.rst", "label": "4 MNm", "rot": 4.0},
    {"path": "data/case_6MNm.rst", "label": "6 MNm", "rot": 4.0},
]

DEFAULT_NODE_ID_RING: int = 260605
DEFAULT_NODE_ID_SHAFT: int = 2582
UNIT_FACTOR: float = 1.0  # Scale factor to output mm
PLOT_LIMIT_ROTATIONS: float = 4.0
INTERPOLATION_POINTS: int = 700
PLOT_TITLE: str = "TRB Main Bearing Load Variation (0° Angle)"
OUTPUT_DIR: str = "output"

# Visual Theme Configuration
COLORMAP = cm.Blues
MAIN_COLOR_RANGE: Tuple[float, float] = (0.40, 1.00)


# =============================================================================
# 2. THESIS / PUBLICATION PLOT SETTINGS
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'axes.linewidth': 1.2,
})


# =============================================================================
# 3. SYNTHETIC MOCK DATA GENERATOR (FOR DEMO & BENCHMARKING)
# =============================================================================

def generate_mock_dpf_data(
    label: str,
    n_steps: int = 50,
    n_rotations: float = 4.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic time and tangential slip arrays for demonstration.

    Args:
        label (str): Case identifier (e.g., '1 MNm') to scale magnitude.
        n_steps (int): Number of load time steps.
        n_rotations (float): Total simulated shaft rotations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Simulated time array (s) and slip array (mm).
    """
    time_vals = np.linspace(0.0, 1.0, n_steps)
    # Extract numeric magnitude from label string if possible
    try:
        load_mag = float(label.split()[0])
    except (ValueError, IndexError):
        load_mag = 2.0

    # Physical approximation: non-linear accumulation of micro-slip over rotation cycles
    base_slip = 0.05 * load_mag * (time_vals ** 1.5)
    harmonic_oscillation = 0.008 * load_mag * np.sin(2 * np.pi * n_rotations * time_vals)
    synthetic_slip = base_slip + harmonic_oscillation
    
    return time_vals, synthetic_slip


# =============================================================================
# 4. ANSYS DPF EXTRACTION
# =============================================================================

def extract_relative_slip(
    rst_path: str,
    label: str,
    node_id_ring: int,
    node_id_shaft: int,
    unit_factor: float = 1.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extracts relative tangential micro-slip between ring and shaft nodes.

    Calculates local polar angle theta = arctan2(Z, X) and projects Cartesian 
    displacements (UX, UZ) into tangential displacements (U_theta):
        U_tangential = -UX * sin(theta) + UZ * cos(theta)
    Relative tangential slip = U_tangential(shaft) - U_tangential(ring).

    Args:
        rst_path (str): File path to ANSYS RST result file.
        label (str): Descriptive label for current load case.
        node_id_ring (int): Node ID on the bearing inner ring surface.
        node_id_shaft (int): Mating node ID on the shaft outer surface.
        unit_factor (float): Multiplier for length unit conversion.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Time array and relative slip array.
    """
    print(f"  Processing: {label}")

    if not os.path.exists(rst_path) or not HAS_ANSYS_DPF:
        if not HAS_ANSYS_DPF:
            print("    [Info] PyMAPDL / DPF library not found in environment.")
        else:
            print(f"    [Info] File not found: '{rst_path}'")
        print("    [Fallback] Using synthetic mock dataset for repository demonstration.")
        return generate_mock_dpf_data(label)

    try:
        sim = post.load_simulation(rst_path)
        mesh = sim.mesh
        all_node_ids = np.array(mesh.node_ids)

        for nid in (node_id_ring, node_id_shaft):
            if nid not in all_node_ids:
                print(f"    [Error] Node ID {nid} not found in RST mesh.")
                return None, None

        all_coords = mesh.coordinates.array
        idx_r = np.where(all_node_ids == node_id_ring)[0][0]
        idx_s = np.where(all_node_ids == node_id_shaft)[0][0]

        # Calculate cylindrical polar coordinate angles in X-Z plane
        theta_r = np.arctan2(all_coords[idx_r][2], all_coords[idx_r][0])
        theta_s = np.arctan2(all_coords[idx_s][2], all_coords[idx_s][0])
        sin_r, cos_r = np.sin(theta_r), np.cos(theta_r)
        sin_s, cos_s = np.sin(theta_s), np.cos(theta_s)

        time_vals = sim.time_freq_support.time_frequencies.data
        slips = []

        for step in range(1, len(time_vals) + 1):
            dr = sim.displacement(node_ids=[node_id_ring], set_ids=[step]).array[0]
            ds = sim.displacement(node_ids=[node_id_shaft], set_ids=[step]).array[0]

            # Project Cartesian displacement vector onto local tangential direction
            tang_r = -dr[0] * sin_r + dr[2] * cos_r
            tang_s = -ds[0] * sin_s + ds[2] * cos_s
            slips.append((tang_s - tang_r) * unit_factor)

        print(f"    ✓ Extracted {len(time_vals)} solution sub-steps")
        return np.array(time_vals), np.array(slips)

    except Exception as err:
        print(f"    [Error] Failed to read RST file via DPF: {err}")
        print("    [Fallback] Generating synthetic mock data instead.")
        return generate_mock_dpf_data(label)


# =============================================================================
# 5. MATHEMATICAL & KINEMATIC TRANSFORMATION
# =============================================================================

def normalize_time(t_raw: np.ndarray, n_rotations: float) -> np.ndarray:
    """Normalises raw solver time step array to total shaft rotations."""
    t_min, t_max = t_raw.min(), t_raw.max()
    if t_max == t_min:
        return np.zeros_like(t_raw)
    return n_rotations * (t_raw - t_min) / (t_max - t_min)


def interpolate_to_grid(
    t_norm: np.ndarray,
    y_raw: np.ndarray,
    common_t: np.ndarray
) -> np.ndarray:
    """Interpolates slip measurements onto a uniform non-dimensional grid."""
    return np.interp(common_t, t_norm, y_raw)


# =============================================================================
# 6. PLOTTING PIPELINE
# =============================================================================

def plot_tangential_slip(
    data: Dict[str, dict],
    common_t: np.ndarray,
    plot_limit: float,
    output_dir: str,
    plot_title: str
) -> str:
    """Generates and exports publication-ready comparative micro-slip plot.

    Args:
        data (Dict[str, dict]): Processed interpolation and visualization dictionary.
        common_t (np.ndarray): Uniform normalized rotation timeline.
        plot_limit (float): Upper x-axis rotation bound.
        output_dir (str): Destination directory for figures.
        plot_title (str): Title stem for exported image file.

    Returns:
        str: Path to saved output image file.
    """
    fig, ax = plt.subplots(figsize=(6.3, 4.0))

    for label, d in data.items():
        ax.plot(
            common_t,
            d["y_interp"],
            label=f"{label}",
            linewidth=2.0,
            color=d["color"],
        )

    # Formatting and styling
    ax.set_xlabel("Shaft Rotation [°]", fontsize=12)
    ax.set_ylabel("Tangential Micro-Slip [mm]", fontsize=12)

    ax.grid(True, which='major', linestyle=':', linewidth=0.6, alpha=0.6, color='gray')
    ax.set_axisbelow(True)

    ax.tick_params(direction='in', length=4, which='major')
    ax.tick_params(direction='in', length=2, which='minor')
    ax.minorticks_on()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Convert normalized rotation factor to angular degrees (1 full rotation = 360°)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: int(x * 360))
    )
    ax.set_xlim(0, plot_limit)

    ax.legend(
        loc='best',
        frameon=True,
        framealpha=0.9,
        edgecolor='gray',
        fancybox=False,
        fontsize=9,
        title="Bending Load",
        title_fontsize=9
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)
    file_stem = plot_title.lower().replace(" ", "_").replace("°", "deg").replace("(", "").replace(")", "")
    output_path = os.path.join(output_dir, f"{file_stem}.png")

    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


# =============================================================================
# 7. MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract and plot relative tangential slip from ANSYS DPF result files."
    )
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for figure exports.")
    args = parser.parse_args()

    t_start = time.perf_counter()
    common_t = np.linspace(0, PLOT_LIMIT_ROTATIONS, INTERPOLATION_POINTS)
    data = {}

    n_cases = len(DEFAULT_CASES)
    main_colors = COLORMAP(np.linspace(*MAIN_COLOR_RANGE, n_cases))

    print("=" * 60)
    print("STEP 1 — Extraction (ANSYS DPF / Synthetic Fallback)")
    print("=" * 60)

    for i, case in enumerate(DEFAULT_CASES):
        t_raw, y_raw = extract_relative_slip(
            case["path"],
            case["label"],
            DEFAULT_NODE_ID_RING,
            DEFAULT_NODE_ID_SHAFT,
            UNIT_FACTOR
        )

        if t_raw is None:
            sys.exit(f"\n[Fatal] Extraction failed for case '{case['label']}'. Aborting.")

        data[case["label"]] = {
            "t_raw": t_raw,
            "y_raw": y_raw,
            "n_rot": case["rot"],
            "color": main_colors[i],
        }

    print("\n" + "=" * 60)
    print("STEP 2 — Normalization & Kinematic Interpolation")
    print("=" * 60)

    for label, d in data.items():
        t_norm = normalize_time(d["t_raw"], d["n_rot"])
        y_interp = interpolate_to_grid(t_norm, d["y_raw"], common_t)
        d["y_interp"] = y_interp
        print(f"  ✓ {label}: Normalised and interpolated to common grid")

    print("\n" + "=" * 60)
    print("STEP 3 — Generating Thesis-Grade Visualizations")
    print("=" * 60)

    saved_path = plot_tangential_slip(
        data, common_t, PLOT_LIMIT_ROTATIONS, args.output_dir, PLOT_TITLE
    )

    print(f"\n  [Success] Saved figure to: {saved_path}")
    print(f"  Total Execution Time: {time.perf_counter() - t_start:.2f} s\n")


if __name__ == "__main__":
    main()
