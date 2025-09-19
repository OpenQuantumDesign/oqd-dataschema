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
from typing import Any, Dict, Literal, Optional, Type, Union

import numpy as np
from bidict import bidict
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
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


class Group(BaseModel, extra="forbid"):
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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__annotations__["class_"] = Literal[cls.__name__]
        setattr(cls, "class_", cls.__name__)

        # Auto-register new group types
        GroupRegistry.register(cls)

    @model_validator(mode="before")
    @classmethod
    def auto_assign_class(cls, data):
        if isinstance(data, BaseModel):
            return data
        if isinstance(data, dict):
            data["class_"] = cls.__name__
        return data


class Dataset(BaseModel, extra="forbid"):
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


class GroupRegistry:
    """Registry for managing group types dynamically"""

    _types: dict[str, Type[Group]] = {}
    _union_cache = None

    @classmethod
    def register(cls, group_type: Type[Group]):
        """Register a new group type"""
        import warnings

        type_name = group_type.__name__

        # Check if type is already registered
        if type_name in cls._types:
            existing_type = cls._types[type_name]
            if existing_type is not group_type:  # Different class with same name
                warnings.warn(
                    f"Group type '{type_name}' is already registered. "
                    f"Overwriting {existing_type} with {group_type}.",
                    UserWarning,
                    stacklevel=2,
                )

        cls._types[type_name] = group_type
        cls._union_cache = None  # Invalidate cache

    @classmethod
    def get_union(cls):
        """Get the current Union of all registered types"""
        if cls._union_cache is None:
            if not cls._types:
                raise ValueError("No group types registered")

            type_list = list(cls._types.values())
            if len(type_list) == 1:
                cls._union_cache = type_list[0]
            else:
                cls._union_cache = Union[tuple(type_list)]

        return cls._union_cache

    @classmethod
    def get_adapter(cls):
        """Get TypeAdapter for current registered types"""
        from typing import Annotated

        union_type = cls.get_union()
        return TypeAdapter(Annotated[union_type, Field(discriminator="class_")])

    @classmethod
    def clear(cls):
        """Clear all registered types (useful for testing)"""
        cls._types.clear()
        cls._union_cache = None

    @classmethod
    def list_types(cls):
        """List all registered type names"""
        return list(cls._types.keys())