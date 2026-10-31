"""Machine-readable capability boundaries; no implicit backend substitution."""
CAPABILITIES = {
    "package": "ariadne-xtal", "version": "0.1.0", "native_full_shelx_parity": False,
    "native": {
        "crystal_geometry": "General positive-definite cells; exact periodic closest image",
        "symmetry": "All Gemmi space-group settings and validated explicit operation groups",
        "scattering": "Independent IT92 kinematic X-ray sum; explicit f-prime/f-double-prime; physical special-site occupancies",
        "phasing": "Seeded multistart P1 charge flipping or sparse atomic projection with supplied cell composition",
        "spacegroup_diagnostics": "Candidate systematic-absence diagnostics; not integrated phase-based space-group identification",
        "refinement": "All work F-squared, scale, positions, isotropic/anisotropic ADPs, optional occupancy and merohedral fractions",
        "constraints": "Site symmetry, continuous origin gauge, native equality/occupancy constraints; see API schema",
        "restraints": "Native distance, angle, similar-distance, planarity, ADP-similarity and rigid-bond residuals",
        "twins": "Fully overlapping integer reciprocal laws preserving metric; identity fraction explicit",
        "absolute_structure": "Inversion-twin refinement and fixed-model weighted Friedel-quotient diagnostic",
        "maps": "Unweighted Fo-Fc, 2Fo-Fc, calculated density, Patterson, CCP4/NPZ export",
        "graphics": "Offline 3D ellipsoids and density surfaces, reciprocal space, diagnostics, SVG/PDF output",
        "agent": "Python API, CLI, structured JSONL and optional official-SDK MCP; isolated manifests and explicit backend",
        "feedback": "Cross-fitted Gaussian profile-deblending research primitive; no raw-image integration pipeline",
    },
    "legacy_bridge": {
        "engines": ["shelxt", "shelxl"], "official_binaries_bundled": False,
        "requires": "Explicit path to a separately installed executable; bridge enabled at agent startup",
        "semantics": "Original INS/HKL/auxiliary bytes passed through; official engine executes its own features",
        "validation": "Transport tested with mock executables; official numerical parity not measured in this environment",
    },
    "not_native": [
        "Complete SHELXT integrated Laue/space-group/origin/element assignment pipeline",
        "Full SHELXL instruction grammar execution, AFIX/HFIX/riding H, FRAG and rigid-group models",
        "General non-merohedral HKLF5 overlap groups",
        "Neutron refinement, extinction, solvent masks/ABIN, SWAT and automatic absorption correction",
        "CGLS/block least-squares, all SHELXL weighting modes and complete CIF archive/checkCIF certification",
        "Raw detector-image indexing/integration/scaling or production instrument control",
        "Analytical or autodifferentiated full refinement Jacobian (native solver uses central finite differences)",
    ],
}


def capabilities():
    import copy
    return copy.deepcopy(CAPABILITIES)
