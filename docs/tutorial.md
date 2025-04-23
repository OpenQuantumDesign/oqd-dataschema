
# Tutorial

```python
import pathlib

import numpy as np
from rich.pretty import pprint

from oqd_dataschema.base import Dataset
from oqd_dataschema.datastore import Datastore
from oqd_dataschema.groups import (
    ExpectationValueDataGroup,
    MeasurementOutcomesDataGroup,
    SinaraRawDataGroup,
)
```

```python
raw = SinaraRawDataGroup(
    camera_images=Dataset(shape=(3, 2, 2), dtype="float32"),
    attrs={"date": "2025-03-26", "version": 0.1},
)
pprint(raw)
```



```python
raw.camera_images.data = np.random.uniform(size=(3, 2, 2)).astype("float32")
pprint(raw)
```



```python
raw.camera_images.data = np.random.uniform(size=(3, 2, 2)).astype("float32")
```



```python
data = Datastore(groups={"raw": raw})
pprint(data)
```




```python
def process_raw(raw: SinaraRawDataGroup) -> MeasurementOutcomesDataGroup:
    processed = MeasurementOutcomesDataGroup(
        outcomes=Dataset(
            data=np.round(raw.camera_images.data.mean(axis=(1, 2))),
        )
    )
    return processed


processed = process_raw(data.groups["raw"])
pprint(processed)
```




```python
data.groups.update(processed=processed)
pprint(data)
```




```python
def process_outcomes(
    measurements: MeasurementOutcomesDataGroup,
) -> ExpectationValueDataGroup:
    expval = ExpectationValueDataGroup(
        expectation_value=Dataset(
            shape=(),
            dtype="float32",
            data=measurements.outcomes.data.mean(),
            attrs={"date": "20", "input": 10},
        )
    )
    return expval


expval = process_outcomes(processed)
data.groups.update(expval=process_outcomes(data.groups["processed"]))

pprint(expval)
```



```python
filepath = pathlib.Path("test.h5")
data.model_dump_hdf5(filepath)
```



```python
data_reload = Datastore.model_validate_hdf5(filepath)
pprint(data_reload)
```