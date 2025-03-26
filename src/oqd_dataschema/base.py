# %%
from typing import Literal, Optional, Union, Any

import numpy as np
from bidict import bidict
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, model_validator, ValidationError, field_validator


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


class Group(BaseModel):
    attrs: Optional[dict[str, Union[int, float, str, complex]]] = {}
    pass


class Dataset(BaseModel):
    dtype: Optional[Literal[tuple(mapping.keys())]] = None
    shape: Optional[tuple[int, ...]] = None
    data: Optional[Any] = Field(default=None, exclude=True)

    attrs: Optional[dict[str, Union[int, float, str, complex]]] = {}

    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        validate_assignment=True
    )

    @model_validator(mode="before")
    @classmethod
    def validate_and_update(cls, values: dict):

        data = values.get("data")
        dtype = values.get('dtype')
        shape = values.get('shape')
        
        if data is None and (dtype is not None and shape is not None):
            return values
        
        elif data is not None and (dtype is None and shape is None):        
            if not isinstance(data, np.ndarray):
                raise TypeError("`data` must be a numpy.ndarray.")

            if data.dtype not in mapping.values():
                raise TypeError(
                    f"`data` must be a numpy array of dtype in {tuple(mapping.keys())}."
                )

            
            values["dtype"] = mapping.inverse[data.dtype]
            values["shape"] = data.shape

        return values

        # else: 
        #     assert data.dtype == dtype and data.shape == shape
        
        # else:
        #     raise ValueError("Must provide either `dtype` and `shape` or `data`.")

    @model_validator(mode="after")
    def validate_data_matches_shape_dtype(self):
        """Ensure that `data` matches `dtype` and `shape`."""
        if self.data is not None:
            expected_dtype = mapping[self.dtype]
            if self.data.dtype != expected_dtype:
                raise ValueError(f"Expected data dtype `{self.dtype}`, but got `{self.data.dtype.name}`.")
            if self.data.shape != self.shape:
                raise ValueError(f"Expected shape {self.shape}, but got {self.data.shape}.")
        return self
    