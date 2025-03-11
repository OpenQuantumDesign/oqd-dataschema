# %%
from typing import Any, Literal, Optional, Union

from rich.pretty import pprint
import h5py
import numpy as np
from bidict import bidict
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
    FilePath
)

# %%
mapping = bidict(
    {
        "int32": np.dtype("int32"),
        "int64": np.dtype("int64"),
        "float32": np.dtype("float32"),
        "float64": np.dtype("float64"),
        "complex64": np.dtype("complex64"),
        "complex128": np.dtype("complex128"),
        # 'string': np.type
    }
)
mapping["int32"]
mapping.inverse[np.dtype("int32")]


# %%
class Dataset(BaseModel):
    model_config = ConfigDict(validate_assignment=True)  # Enable assignment validation
    
    dtype: Optional[Literal[tuple(mapping.keys())]] = None
    shape: Optional[tuple[int, ...]] = None
    data: Optional[Any] = Field(None, exclude=True)
    _path: Optional[str] = None
    # todo: add validation that its a numpy array

    @model_validator(mode="before")
    @classmethod
    def validate_and_update(cls, values: dict):
        data = values.get("data")
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("`data` must be a numpy.ndarray.")

            if data.dtype not in mapping.values():
                raise TypeError(
                    f"`data` must be a numpy array of dtype in {tuple(mapping.keys())}."
                )

            values["dtype"] = mapping.inverse[data.dtype]
            values["shape"] = data.shape

        return values
    

class Group(BaseModel):
    model_config = ConfigDict(extra='allow') 
     
    # datasets: dict[str, Dataset]

    # def __getitem__(self, key):
    #     if self.datasets[key] is None and self.datasets[key]._path is not None:
            
        # return self.datasets[key]
    

class DataSchema(BaseModel):
    groups: dict[str, Group]
    _filename: Optional[FilePath] = None
    
    def dump(self, filename):
        with h5py.File(filename, "w") as f:
            f.attrs["json"] = dataschema.model_dump_json()
            for gname, group in dataschema.groups.items():
                h5group = f.create_group(gname)
                for dname, dataset in group.datasets.items():
                    print(gname, dname)
                    h5group.create_dataset(dname, data=dataset.data)

    @classmethod
    def load(cls, filename):
        with h5py.File(filename, "r") as f:
            self = DataSchema.model_validate_json(f.attrs['json'])

        self._filename = filename
        for gname, group in self.groups.items():
            for dname, dataset in group.datasets.items():
                print(gname, dname)
                dataset._path = f"{gname}/{dname}"
        return self
    
        
        
dataspec = Dataset(
    data=np.ones(shape=[10, 12], dtype=np.int64),
    # dtype="int64",
    # shape=(10, 12),
    _path='as'
)
dataspec.data = np.ones(shape=[4, 12], dtype=np.float32)
pprint(dataspec)
dataspec._path = 'as'
pprint(dataspec._path)

# print(dataspec.shape)

# %%
# raw_data_group = Group(
#     datasets={
#         "camera_images": Dataset(data=np.ones([10, 10], np.complex128)),
#         "photodetector": Dataset(data=np.ones([10, 10], np.complex128)),
#     }
# )

raw_data_group = Group(
    "camera_images"=Dataset(data=np.ones([10, 10], np.complex128)),
    "photodetector"=Dataset(data=np.ones([10, 10], np.complex128)),
)
# print(raw_data_group)
print(raw_data_group.camera_images)

# %%
dataschema = DataSchema(
    groups={
        "raw": Group(
            datasets={
                "camera_images": Dataset(data=np.ones([10, 10], np.float64)),
                "photodetector": Dataset(data=np.ones([10, 10], np.int32)),
            }
        ),
        "pipeline": Group(
            datasets={
                "states": Dataset(data=np.ones([1, 50], np.complex128)),
            }
        ),
    }
)
pprint(dataschema)
json = dataschema.model_dump_json()
pprint(json)

# %%
dataschema.dump('data.h5')

# %%
f = h5py.File("data.h5", "r")

#%%
ds = DataSchema.load('data.h5')



#%%
json = DataSchema.model_validate_json(f.attrs['json'])
pprint(json)

f['raw']['camera_images'][()]

f['raw/camera_images'][()]

#%%
f.close()

#%%
class Metrics(Group):
    datasets: dict[str, Dataset]

class Counts(Group):
    datasets: dict[str, Dataset]
    
class AnalogResult(Group):
    times: Dataset
    # states: Dataset
    # metrics: Metrics
    # counts: Counts


counts = Counts(datasets={'01': Dataset(data=np.array(10,))})
metrics = Metrics(datasets={'entropy': Dataset(data=np.linspace(0, 10, 25))})

result = AnalogResult(
    times=Dataset(data=np.ones([19, 10])),
    # states=Dataset(data=np.ones([19, 10])),
    # metrics=metrics,
    # counts=counts
)

#%%
with h5py.File("data.h5", "r") as f:
    sch = DataSchema.model_validate_json(f.attrs['json'])
    
    # groups = {}
    # for gname, h5group in f.items():
    #     datasets = {}
    #     for dname, dset in h5group.items():
    #         data = dset[()]
    #         dtype = mapping.inverse[data.dtype]
    #         shape = data.shape
    #         datasets[dname] = Dataset(data=data, dtype=dtype, shape=shape)
    #     groups[gname] = Group(datasets=datasets)
    # dataschema = DataSchema(groups=groups)
    # pprint(dataschema)



# %%
