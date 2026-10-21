# ANSYS DPF Wind Turbine Main Bearing Ring Creep Analysis

An automated Python post-processing pipeline for ANSYS FEA simulation result files (`.rst`). This tool utilizes the ANSYS Data Processing Framework (ANSYS DPF) to extract relative tangential micro-slip between mating sub-assembly components, normalise solver time steps to total shaft rotation angles, and produce publication-quality comparative plots across parameterized bending load conditions.

---

## Technical Overview

In wind turbine drivetrain main bearings, cyclic multi-axial loads (radial forces, thrust, and tilting moments) induce localized elastic deformation in the bearing rings. Under heavy operational bending moments, micro-slip occurs at the interference fit interface between the inner ring and shaft, leading to tangential ring creep ($\text{RC}_{\text{rate}}$).

This pipeline programmatically evaluates micro-slip at mating node pairs across parameterized load steps (e.g., 1 MNm to 6 MNm) by projecting Cartesian nodal displacements onto local polar coordinate tangents:

```math
\Delta u_\theta = u_{\theta, \text{shaft}} - u_{\theta, \text{ring}} = (-u_{x,s}\sin\theta_s + u_{z,s}\cos\theta_s) - (-u_{x,r}\sin\theta_r + u_{z,r}\cos\theta_r)

flowchart TD
    A[ANSYS RST Result Files] --> B{ANSYS DPF Installed?}
    B -- Yes --> C[Load DPF Simulation Context]
    B -- No / Missing RST --> D[Trigger Synthetic Mock Data Generator]
    
    C --> E[Extract Coordinates & Nodal IDs]
    E --> F[Calculate Polar Angles θ_ring & θ_shaft]
    F --> G[Iterate Time Steps: Extract UX, UZ Displacements]
    G --> H[Project Displacements onto Tangential Vector]
    
    D --> I[Simulate Time & Slip Vectors]
    H --> I
    
    I --> J[Normalise Solver Time to Rotation Degrees]
    J --> K[Interpolate Slip onto Uniform Grid]
    K --> L[Generate Thesis-Grade Matplotlib Visualization]
    L --> M[Export Figure to /output Directory]
