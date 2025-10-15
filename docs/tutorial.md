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
    t: Dataset
    x: Dataset
```

Defined groups are automatically registered into the [`GroupRegistry`][oqd_dataschema.group.GroupRegistry].

```python
from oqd_dataschema import GroupRegistry

GroupRegistry.groups
```

## Initialize Group

```python
t = np.linspace(0, 1, 101).astype(np.float32)
x = np.sin(t).astype(np.complex64)

group = CustomGroup(
    t=Dataset(dtype="float32", shape=(101,)), x=Dataset(dtype="complex64", shape=(101,))
)

group.t.data = t
group.x.data = x
```

## Initialize Datastore

```python
from oqd_datastore import Datastore

datastore = Datastore(groups={"g1": group})
```

## Data pipeline

```python
def process(datastore) -> Datastore:
    _g = datastore.get("g1")

    g2 = CustomGroup(t=Dataset(data=_g.t.data), x=Dataset(data=_g.x.data + 1j))

    datastore.add(g2=g2)

    return datastore


datastore.pipe(process)
```

## Save Datastore

```python
datastore.model_dump_hdf5(pathlib.Path("datastore.h5"), mode="w")
```

## Load Datastore

```python
reloaded_datastore = Datastore.model_validate_hdf5(pathlib.Path("datastore.h5"))
```
