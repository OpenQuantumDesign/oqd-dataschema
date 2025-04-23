# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
from typing import Any, Literal, Optional, Union

import numpy as np
from bidict import bidict
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
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


class Group(BaseModel):
    """
    Schema representation for a group object within an HDF5 file.
    
    Each grouping of data should be defined as a subclass of `Group`, and specify the datasets that it will contain. 
    This base object only has attributes, `attrs`, which are associated to the HDF5 group.
    
    Attributes:
        attrs: A dictionary of attributes to append to the dataset.

    Example:
        ```
        group = Group(attrs={'version': 2, 'date': '2025-01-01'})
        ```
    """
    
    attrs: Optional[dict[str, Union[int, float, str, complex]]] = {}


class Dataset(BaseModel):
    """
    Schema representation for a dataset object to be saved within an HDF5 file.
    
    Attributes:
        dtype: The datatype of the dataset, such as `int32`, `float32`, `int64`, `float64`, etc. 
            Types are inferred from the `data` attribute if provided.
        shape: The shape of the dataset.
        data: The numpy ndarray of the data, from which `dtype` and `shape` are inferred.
        
        attrs: A dictionary of attributes to append to the dataset.

    Example:
        ```
        dataset = Dataset(data=np.array([1, 2, 3, 4]))
        
        dataset = Dataset(dtype='int64', shape=[4,])
        dataset.data = np.array([1, 2, 3, 4])
        ```
    """
    dtype: Optional[Literal[tuple(mapping.keys())]] = None
    shape: Optional[tuple[int, ...]] = None
    data: Optional[Any] = Field(default=None, exclude=True)

    attrs: Optional[dict[str, Union[int, float, str, complex]]] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def validate_and_update(cls, values: dict):
        data = values.get("data")
        dtype = values.get("dtype")
        shape = values.get("shape")

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
                raise ValueError(
                    f"Expected data dtype `{self.dtype}`, but got `{self.data.dtype.name}`."
                )
            if self.data.shape != self.shape:
                raise ValueError(
                    f"Expected shape {self.shape}, but got {self.data.shape}."
                )
        return self
