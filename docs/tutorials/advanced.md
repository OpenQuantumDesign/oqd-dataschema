# Tutorial

## Group Definition

```python
from oqd_dataschema import GroupBase, Attrs

class CustomGroup(GroupBase):
    attrs: Attrs = Field(
        default_factory=lambda: dict(
            timestamp=str(datetime.datetime.now(datetime.timezone.utc))
        )
    )
    dset: Dataset
    tbl: Table
    fld: Folder
```

Defined groups are automatically registered into the [`GroupRegistry`][oqd_dataschema.group.GroupRegistry].

```python
from oqd_dataschema import GroupRegistry

GroupRegistry.groups
```

## Initialize Group

```python
from oqd_dataschema import Dataset, Table, Folder, unstructured_to_structured

dset = Dataset(data=np.linspace(0, 1, 101).astype(np.float32))
tbl = Table(
    columns=[("t", "float32"), ("x", "complex128")],
    data=unstructured_to_structured(
        np.stack([np.linspace(0, 1, 101), np.sin(np.linspace(0, 1, 101))], -1),
        dtype=np.dtype([("t", np.float32), ("x", np.complex128)]),
    ),
)
fld = Folder(
    document_schema={"t": "float32", "signal": {"x": "complex128", "y": "complex128"}},
    data=unstructured_to_structured(
        np.stack(
            [
                np.linspace(0, 1, 101),
                np.sin(np.linspace(0, 1, 101)),
                np.cos(np.linspace(0, 1, 101)),
            ],
            -1,
        ),
        dtype=np.dtype(
            [
                ("t", np.float32),
                ("signal", np.dtype([("x", np.complex128), ("y", np.complex128)])),
            ]
        ),
    ),
)


group = CustomGroup(dset=dset, tbl=tbl, fld=fld)
```

## Initialize Datastore

```python
from oqd_datastore import Datastore

datastore = Datastore(groups={"g1": group})
```

## Save Datastore

```python
datastore.model_dump_hdf5(pathlib.Path("datastore.h5"), mode="w")
```

## Load Datastore

```python
reloaded_datastore = Datastore.model_validate_hdf5(pathlib.Path("datastore.h5"))
```
