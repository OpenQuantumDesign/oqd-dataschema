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
import warnings
from typing import Annotated, Any, ClassVar, Literal, Optional, Union

import numpy as np
from bidict import bidict
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    TypeAdapter,
    model_validator,
)

########################################################################################

__all__ = ["GroupBase", "Dataset", "GroupRegistry"]

########################################################################################


invalid_attrs = ["_model_signature", "_model_json"]


def _valid_attr_key(value):
    if value in invalid_attrs:
        raise KeyError

    return value


Attrs = Optional[
    dict[
        Annotated[str, BeforeValidator(_valid_attr_key)],
        Union[int, float, str, complex],
    ]
]


# %%
dtype_map = bidict(
    {
        "int16": np.dtypes.Int16DType,
        "int32": np.dtypes.Int32DType,
        "int64": np.dtypes.Int64DType,
        "float16": np.dtypes.Float16DType,
        "float32": np.dtypes.Float32DType,
        "float64": np.dtypes.Float64DType,
        "complex64": np.dtypes.Complex64DType,
        "complex128": np.dtypes.Complex128DType,
        "string": np.dtypes.StrDType,
        "bytes": np.dtypes.BytesDType,
        "bool": np.dtypes.BoolDType,
    }
)


class GroupBase(BaseModel, extra="forbid"):
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

    attrs: Attrs = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for k, v in cls.__annotations__.items():
            if k == "class_":
                raise AttributeError("`class_` attribute should not be set manually.")

            if k == "attrs" and k is not Attrs:
                raise TypeError("`attrs` should be of type `Attrs`")

            if k not in ["class_", "attrs"] and v not in [Dataset, ClassVar]:
                raise TypeError(
                    "All fields of `GroupBase` have to be of type `Dataset`."
                )

        cls.__annotations__["class_"] = Literal[cls.__name__]
        setattr(cls, "class_", cls.__name__)

        # Auto-register new group types
        GroupRegistry.register(cls)


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

    dtype: Optional[Literal[tuple(dtype_map.keys())]] = None
    shape: Optional[tuple[int, ...]] = None
    data: Optional[Any] = Field(default=None, exclude=True)

    attrs: Attrs = {}

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

            if type(data.dtype) not in dtype_map.values():
                raise TypeError(
                    f"`data` must be a numpy array of dtype in {tuple(dtype_map.keys())}."
                )

            values["dtype"] = dtype_map.inverse[type(data.dtype)]
            values["shape"] = data.shape

        return values

    @model_validator(mode="after")
    def validate_data_matches_shape_dtype(self):
        """Ensure that `data` matches `dtype` and `shape`."""
        if self.data is not None:
            expected_dtype = dtype_map[self.dtype]
            if type(self.data.dtype) is not expected_dtype:
                raise ValueError(
                    f"Expected data dtype `{self.dtype}`, but got `{self.data.dtype.name}`."
                )
            if self.data.shape != self.shape:
                raise ValueError(
                    f"Expected shape {self.shape}, but got {self.data.shape}."
                )
        return self


class MetaGroupRegistry(type):
    def __new__(cls, clsname, superclasses, attributedict):
        attributedict["groups"] = dict()
        return super().__new__(cls, clsname, superclasses, attributedict)

    def register(cls, group):
        if not issubclass(group, GroupBase):
            raise TypeError("You may only register subclasses of GroupBase.")

        if group.__name__ in cls.groups.keys():
            warnings.warn(
                f"Overwriting previously registered `{group.__name__}` group of the same name.",
                UserWarning,
                stacklevel=2,
            )

        cls.groups[group.__name__] = group

    def clear(cls):
        """Clear all registered types (useful for testing)"""
        cls.groups.clear()

    @property
    def union(cls):
        """Get the current Union of all registered types"""
        return Annotated[
            Union[tuple(cls.groups.values())], Discriminator(discriminator="class_")
        ]

    @property
    def adapter(cls):
        """Get TypeAdapter for current registered types"""
        return TypeAdapter(cls.union)


class GroupRegistry(metaclass=MetaGroupRegistry):
    pass
