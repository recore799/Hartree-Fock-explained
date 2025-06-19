import os

# Redefine base path after code execution state reset
base_path = "/mnt/data/HF"

# Define folders and files again
folders = [
    "HF/diat",
    "HF/demo"
]

files = {
    "HF/README.md": "# Hartree-Fock Project\n\nPrototype implementation of a general SCF method for H₂ using STO-3G.",
    "HF/.gitignore": "__pycache__/\n*.pyc\n.ipynb_checkpoints/\n.env/\nvenv/",
    "HF/requirements.txt": "numpy\nscipy\nmatplotlib",
    "HF/diat/__init__.py": "",
    "HF/diat/scf.py": "# scf.py\n\n# Placeholder for the SCF routine\ndef run_scf():\n    print(\"Running SCF procedure (stub)\")",
    "HF/diat/basis.py": "# basis.py\n\n# Placeholder for basis set generation\ndef make_sto3g(atom):\n    print(f\"Generating STO-3G basis for {atom} (stub)\")",
    "HF/diat/integrals.py": "# integrals.py\n\n# Placeholder for integral calculations\ndef compute_integrals():\n    print(\"Computing integrals (stub)\")",
    "HF/diat/utils.py": "# utils.py\n\n# Placeholder for utility functions\ndef build_density():\n    print(\"Building density matrix (stub)\")",
    "HF/demo/H2-scf-demo.ipynb": ""  # Will be created as an empty notebook
}

# Create folders
for folder in folders:
    os.makedirs(f"~/uni/{folder}", exist_ok=True)

# Create files
for path, content in files.items():
    with open(f"/mnt/data/{path}", "w") as f:
        f.write(content)

