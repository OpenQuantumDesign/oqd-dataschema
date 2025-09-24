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
from functools import partial, reduce
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
    model_validator,
)

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

invalid_attrs = ["_datastore_signature", "_group_json"]


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
    shape: Optional[Tuple[int, ...]] = None
    data: Optional[Any] = Field(default=None, exclude=True)

    attrs: Attrs = {}

    model_config = ConfigDict(
        use_enum_values=False, arbitrary_types_allowed=True, validate_assignment=True
    )

    @classmethod
    def cast(cls, data):
        if isinstance(data, np.ndarray):
            return cls(data=data)
        return data

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

            if type(data.dtype) not in DTypes:
                raise TypeError(
                    f"`data` must be a numpy array of dtype in {tuple(DTypes.names())}."
                )

            values["dtype"] = DTypes(type(data.dtype)).name.lower()
            values["shape"] = data.shape

        return values

    @model_validator(mode="after")
    def validate_data_matches_shape_dtype(self):
        """Ensure that `data` matches `dtype` and `shape`."""
        if self.data is not None:
            expected_dtype = DTypes.get(self.dtype).value
            if type(self.data.dtype) is not expected_dtype:
                raise ValueError(
                    f"Expected data dtype `{self.dtype}`, but got `{self.data.dtype.name}`."
                )
            if self.data.shape != self.shape:
                raise ValueError(
                    f"Expected shape {self.shape}, but got {self.data.shape}."
                )
        return self

    def __getitem__(self, idx):
        return self.data[idx]


def _constrain_dtype(dataset, *, dtype_constraint=None):
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

    return dataset


def _constraint_dim(dataset, *, min_dim=None, max_dim=None):
    min_dim = 0 if min_dim is None else min_dim

    dims = len(dataset.shape)

    if dims < min_dim or (max_dim is not None and dims > max_dim):
        raise ValueError(
            f"Expected {min_dim} <= dimension of shape{f' <= {max_dim}'}, but got shape = {dataset.shape}."
        )

    return dataset


def _constraint_shape(dataset, *, shape_constraint=None):
    if shape_constraint and (
        len(shape_constraint) != len(dataset.shape)
        or reduce(
            lambda x, y: x or y,
            map(
                lambda x: x[0] is not None and x[0] != x[1],
                zip(shape_constraint, dataset.shape),
            ),
        )
    ):
        raise ValueError(
            f"Expected shape to be {shape_constraint}, but got {dataset.shape}."
        )

    return dataset


def condataset(
    *, shape_constraint=None, dtype_constraint=None, min_dim=None, max_dim=None
):
    return Annotated[
        Dataset,
        BeforeValidator(Dataset.cast),
        AfterValidator(partial(_constrain_dtype, dtype_constraint=dtype_constraint)),
        AfterValidator(partial(_constraint_dim, min_dim=min_dim, max_dim=max_dim)),
        AfterValidator(partial(_constraint_shape, shape_constraint=shape_constraint)),
    ]


CastDataset = Annotated[Dataset, BeforeValidator(Dataset.cast)]


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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for k, v in cls.__annotations__.items():
            if k == "class_":
                raise AttributeError("`class_` attribute should not be set manually.")

            if k == "attrs" and k is not Attrs:
                raise TypeError("`attrs` should be of type `Attrs`")

            if (
                k not in ["class_", "attrs"]
                and v not in [Dataset, ClassVar]
                and not (typing.get_origin(v) == Annotated and v.__origin__ is Dataset)
            ):
                raise TypeError(
                    "All fields of `GroupBase` have to be of type `Dataset`."
                )

        cls.__annotations__["class_"] = Literal[cls.__name__]
        setattr(cls, "class_", cls.__name__)

        # Auto-register new group types
        GroupRegistry.register(cls)


########################################################################################


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


# %%
