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

To install,
```bash
pip install git+https://github.com/OpenQuantumDesign/oqd-dataschema.git
```
