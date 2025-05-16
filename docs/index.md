# 

<p align="center">
  <img src="img/oqd-logo-black.png#only-light" alt="Logo" style="max-height: 200px;">
  <img src="img/oqd-logo-white.png#only-dark" alt="Logo" style="max-height: 200px;">
</p>

<div align="center">
    <h2 align="center">
    Open Quantum Design: Dataschema
    </h2>
</div>

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<!-- prettier-ignore -->
/// admonition | Note
    type: note

Still in very early stages of development! Significant breaking changes are expected.

///

This library defines a standard dataschema for storing large, processed datasets from OQD classical emulators and real hardware devices.
The design goals are to have:

- A tool which standardizes data into the HDF5 format.
- Minimizes *a priori* knowledge that is needed of the internal heirarchical structure, reducing friction for users to load data.
- Transparently return both raw and processed data, where the levels of post-processing can be selected by the user.

To install,

```bash
pip install git+https://github.com/OpenQuantumDesign/oqd-dataschema.git
```
