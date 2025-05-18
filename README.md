# ![Open Quantum Design](https://raw.githubusercontent.com/OpenQuantumDesign/oqd-core/main/docs/img/oqd-logo-text.png)

<h2 align="center">
    Open Quantum Design: Dataschema
</h2>

> [!NOTE]
> :bangbang: Still in very early stages of development! Significant breaking changes are expected.

This library defines a standard dataschema for storing large, processed datasets from OQD classical emulators and real hardware devices.
The design goals are to have:

- A tool which standardizes data into the HDF5 format.
- Minimizes *a priori* knowledge that is needed of the internal heirarchical structure, reducing friction for users to load data.
- Transparently return both raw and processed data, where the levels of post-processing can be selected by the user.

## Installation

### Using pip
To install using pip:
```bash
pip install git+https://github.com/OpenQuantumDesign/oqd-dataschema.git
```

### Using Nix

If you have Nix package manager with flakes enabled, you can install and use this package directly from the repository:

```bash
# Run in a development shell
nix develop github:OpenQuantumDesign/oqd-dataschema

# Or install the package
nix profile install github:OpenQuantumDesign/oqd-dataschema
```

#### Development Environment

For development, clone the repository and use the provided development shell:

```bash
git clone https://github.com/OpenQuantumDesign/oqd-dataschema.git
cd oqd-dataschema
nix develop
```

This will provide you with a complete development environment including:
- All runtime dependencies (pydantic, h5py, bidict)
- Development tools (black, ruff, mypy)
- Documentation tools (mkdocs with required extensions)

The development shell automatically sets up your PYTHONPATH and provides all necessary tools to build, test, and document the project.
