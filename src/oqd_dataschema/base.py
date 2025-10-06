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
import typing
import warnings
from enum import Enum
from types import NoneType
from typing import Annotated, Any, ClassVar, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from .utils import _flex_shape_equal, _validator_from_condition

########################################################################################

__all__ = ["GroupBase", "Dataset", "GroupRegistry", "condataset", "CastDataset"]

########################################################################################


class DTypes(Enum):
    BOOL = np.dtypes.BoolDType
    INT16 = np.dtypes.Int16DType
    INT32 = np.dtypes.Int32DType
    INT64 = np.dtypes.Int64DType
    UINT16 = np.dtypes.UInt16DType
    UINT32 = np.dtypes.UInt32DType
    UINT64 = np.dtypes.UInt64DType
    FLOAT16 = np.dtypes.Float16DType
    FLOAT32 = np.dtypes.Float32DType
    FLOAT64 = np.dtypes.Float64DType
    COMPLEX64 = np.dtypes.Complex64DType
    COMPLEX128 = np.dtypes.Complex128DType
    STR = np.dtypes.StrDType
    BYTES = np.dtypes.BytesDType
    STRING = np.dtypes.StringDType

    @classmethod
    def get(cls, name):
        return cls[name.upper()]

    @classmethod
    def names(cls):
        return tuple((dtype.name.lower() for dtype in cls))


########################################################################################

invalid_attrs = ["_datastore_signature", "_group_schema"]


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

########################################################################################


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

    dtype: Optional[Literal[DTypes.names()]] = None  # type: ignore
    shape: Optional[Tuple[Union[int, None], ...]] = None
    data: Optional[Any] = Field(default=None, exclude=True)

    attrs: Attrs = {}

    model_config = ConfigDict(
        use_enum_values=False, arbitrary_types_allowed=True, validate_assignment=True
    )

    @field_validator("data", mode="before")
    @classmethod
    def validate_and_update(cls, value):
        # check if data exist
        if value is None:
            return value

        # check if data is a numpy array
        if not isinstance(value, np.ndarray):
            raise TypeError("`data` must be a numpy.ndarray.")

        return value

    @model_validator(mode="after")
    def validate_data_matches_shape_dtype(self):
        """Ensure that `data` matches `dtype` and `shape`."""

        # check if data exist
        if self.data is None:
            return self

        # check if dtype matches data
        if (
            self.dtype is not None
            and type(self.data.dtype) is not DTypes.get(self.dtype).value
        ):
            raise ValueError(
                f"Expected data dtype `{self.dtype}`, but got `{self.data.dtype.name}`."
            )

        # check if shape mataches data
        if self.shape is not None and not _flex_shape_equal(
            self.data.shape, self.shape
        ):
            raise ValueError(f"Expected shape {self.shape}, but got {self.data.shape}.")

        # reassign dtype if it is None
        if self.dtype != DTypes(type(self.data.dtype)).name.lower():
            self.dtype = DTypes(type(self.data.dtype)).name.lower()

        # resassign shape to concrete value if it is None or a flexible shape
        if self.shape != self.data.shape:
            self.shape = self.data.shape

        return self

    @classmethod
    def cast(cls, data):
        if isinstance(data, np.ndarray):
            return cls(data=data)
        return data

    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def _is_dataset_type(cls, type_):
        return type_ == cls or (
            typing.get_origin(type_) is Annotated and type_.__origin__ is cls
        )


CastDataset = Annotated[Dataset, BeforeValidator(Dataset.cast)]

########################################################################################


@_validator_from_condition
def _constrain_dtype(dataset, *, dtype_constraint=None):
    """Constrains the dtype of a dataset"""
    if (not isinstance(dtype_constraint, str)) and isinstance(
        dtype_constraint, Sequence
    ):
        dtype_constraint = set(dtype_constraint)
    elif isinstance(dtype_constraint, str):
        dtype_constraint = {dtype_constraint}

    if dtype_constraint and dataset.dtype not in dtype_constraint:
        raise ValueError(
            f"Expected dtype to be of type one of {dtype_constraint}, but got {dataset.dtype}."
        )


@_validator_from_condition
def _constraint_dim(dataset, *, min_dim=None, max_dim=None):
    """Constrains the dimension of a dataset"""
    if min_dim is not None and max_dim is not None and min_dim > max_dim:
        raise ValueError("Impossible to satisfy dimension constraints on dataset.")

    min_dim = 0 if min_dim is None else min_dim

    dims = len(dataset.shape)

    if dims < min_dim or (max_dim is not None and dims > max_dim):
        raise ValueError(
            f"Expected {min_dim} <= dimension of shape{f' <= {max_dim}'}, but got shape = {dataset.shape}."
        )


@_validator_from_condition
def _constraint_shape(dataset, *, shape_constraint=None):
    """Constrains the shape of a dataset"""
    if shape_constraint and not _flex_shape_equal(shape_constraint, dataset.shape):
        raise ValueError(
            f"Expected shape to be {shape_constraint}, but got {dataset.shape}."
        )


def condataset(
    *, shape_constraint=None, dtype_constraint=None, min_dim=None, max_dim=None
):
    """Implements dtype, dimension and shape constrains on the dataset."""
    return Annotated[
        CastDataset,
        AfterValidator(_constrain_dtype(dtype_constraint=dtype_constraint)),
        AfterValidator(_constraint_dim(min_dim=min_dim, max_dim=max_dim)),
        AfterValidator(_constraint_shape(shape_constraint=shape_constraint)),
    ]


########################################################################################


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

    @classmethod
    def _is_allowed_field_type(cls, v):
        is_dataset = Dataset._is_dataset_type(v)

        is_annotated_dataset = typing.get_origin(
            v
        ) is Annotated and Dataset._is_dataset_type(v.__origin__)

        is_optional_dataset = typing.get_origin(v) is Union and (
            (v.__args__[0] == NoneType and Dataset._is_dataset_type(v.__args__[1]))
            or (v.__args__[1] == NoneType and Dataset._is_dataset_type(v.__args__[0]))
        )

        is_dict_dataset = (
            typing.get_origin(v) is dict
            and v.__args__[0] is str
            and Dataset._is_dataset_type(v.__args__[1])
        )

        return (
            is_dataset or is_annotated_dataset or is_optional_dataset or is_dict_dataset
        )

    @classmethod
    def _is_classvar(cls, v):
        return v is ClassVar or typing.get_origin(v) is ClassVar

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for k, v in cls.__annotations__.items():
            if k in ["class_", "attrs"]:
                raise AttributeError(
                    "`class_` and `attrs` attribute should not be set manually."
                )

            if cls._is_classvar(v):
                continue

            if not cls._is_allowed_field_type(v):
                raise TypeError(
                    "All fields of `GroupBase` have to be of type `Dataset`."
                )

        cls.__annotations__["class_"] = Literal[cls.__name__]
        setattr(cls, "class_", cls.__name__)

        # Auto-register new group types
        GroupRegistry.register(cls)


########################################################################################


class MetaGroupRegistry(type):
    """
    Metaclass for the GroupRegistry
    """

    def __new__(cls, clsname, superclasses, attributedict):
        attributedict["groups"] = dict()
        return super().__new__(cls, clsname, superclasses, attributedict)

    def register(cls, group):
        """Registers a group into the GroupRegistry."""
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
    """
    Represents the GroupRegistry
    """

    pass


# %%
