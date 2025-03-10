#%%
from pydantic import BaseModel, FilePath,PrivateAttr, ConfigDict, model_validator, Field
from pydantic import ValidationError
from typing import Literal, Optional, Any
import numpy as np
from bidict import bidict
import h5py

#%%
mapping = bidict(
    {
        'int32': np.dtype('int32'),
        'int64': np.dtype('int64'),
        'float32': np.dtype('float32'),
        'float64': np.dtype('float64'),
        'complex64': np.dtype('complex64'),
        'complex128': np.dtype('complex128'),
        # 'string': np.type
    }
)
mapping['int32']
mapping.inverse[np.dtype('int32')]

#%%
class Dataset(BaseModel):
    # model_config = ConfigDict(validate_assignment=True, allow_extra=True)
    
    path: str
    # _shape: Optional[tuple[int, ...]] = None
    # _dtype: Optional[Literal[tuple(mapping.keys())]] = None
    # _data: Optional[Any] = None

    dtype: Optional[Literal[tuple(mapping.keys())]] = Field(None, exclude=True)
    shape: Optional[tuple[int, ...]] = Field(None, exclude=True)
    data: Optional[Any] = Field(None, exclude=True)
    
    # _dtype: PrivateAttr(Optional[Literal[tuple(mapping.keys())]]) = None
    # dtype: Optional[Literal[tuple(mapping.keys())]] = Field(alias='_dtype')
    
    def model_post_init(self, __context: Any) -> None:
        print(self.data)
        # if self._data:
            # self._shape = self._data.shape
            # self._dtype = mapping.inverse[self._data.dtype]
        # self._dtype = 'int32'
        
    # class Config:
        # extra = "forbid"  # Prevents unexpected fields in the model

    # @model_validator(mode="before")
    # @classmethod
    # def set_private_attrs(cls, values):
    #     if "dtype" in values:
    #         values["_dtype"] = values.pop("dtype")  # Move "a" to _a internally
    #     return values

    # @property
    # def data(self):
    #     return self._data
    
    # @data.setter
    # def data(self, data: np.ndarray):
    #     if not isinstance(data, np.ndarray):
    #         raise TypeError("`data` must be a numpy.ndarray.")
    #     if data.dtype not in tuple(mapping.values()):
    #         raise TypeError(f"`data` must be a numpy array of dtype in {tuple(mapping.keys())}.")
            
    #     self._data = data
    #     self._shape = data.shape
    #     self._dtype = mapping.inverse[data.dtype]
    #     return 
    
    # @property
    # def dtype(self):
    #     return self._dtype
    
    # @property
    # def shape(self):
    #     return self._shape
    

class DataSchema(BaseModel):
    schemas: dict[str, DataSpec]
    _filename: Optional[FilePath] = None

dataspec = DataSpec(
    path="states", 
    data=np.ones(shape=[10, 12], dtype=np.int64),
    dtype='int64',
    shape=(10, 12),
)
# dataspec.data = np.ones(shape=[10, 12], dtype=np.int64)
print(dataspec.dtype)
# print(dataspec.shape)

#%%

dataschema = DataSchema(
    schemas={
        'states': DataSpec(path="states", data=np.ones(shape=[10, 12], dtype=np.int64)),
        'times': DataSpec(path="times", data=np.ones(shape=[10, 12], dtype=np.int64)),
    }
)

#%%
with h5py.File('data.h5', 'w') as f:
    f.create_dataset('dataset_name', data=arr)


#%%

#%%
print(dataschema.model_dump_json())



# %%
