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
from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, Literal, Union

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
)

########################################################################################

__all__ = ["Attrs", "DTypes", "DTypeNames", "GroupField"]

########################################################################################


class DTypes(Enum):
    """
    Enum for data types supported by oqd-dataschema.

    |Type   |Variant|
    |-------|-------|
    |Boolean|`BOOL` |
    |Integer|`INT16`, `INT32`, `INT64` (signed)<br>`UINT16`, `UINT32`, `UINT64` (unsigned)|
    |Float  |`FLOAT32`, `FLOAT64`|
    |Complex|`COMPLEX64`, `COMPLEX128`|
    |Bytes  |`BYTES`|
    |String |`STR`, `STRING`|
    """

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
    def get(cls, name: str) -> DTypes:
        """
        Get the [`DTypes`][oqd_dataschema.base.DTypes] enum variant by lowercase name.
        """
        return cls[name.upper()]

    @classmethod
    def names(cls):
        """
        Get the lowercase names of all variants of [`DTypes`][oqd_dataschema.base.DTypes] enum.
        """
        return tuple((dtype.name.lower() for dtype in cls))


DTypeNames = Literal[DTypes.names()]
"""
Literal list of lowercase names for [`DTypes`][oqd_dataschema.base.DTypes] variants.
"""


########################################################################################

invalid_attrs = ["_datastore_signature", "_group_schema"]


def _valid_attr_key(value: str) -> str:
    """
    Validates attribute keys (prevents overwriting of protected attrs).
    """
    if value in invalid_attrs:
        raise KeyError

    return value


AttrKey = Annotated[str, BeforeValidator(_valid_attr_key)]
"""
Annotated type that represents a valid key for attributes (prevents overwriting of protected attrs).
"""

Attrs = dict[AttrKey, Union[int, float, str, complex]]
"""
Type that represents attributes of an object.
"""

########################################################################################


class GroupField(BaseModel, ABC):
    """
    Abstract class for a valid data field of Group.

    Attributes:
        attrs: A dictionary of attributes to append to the object.
    """

    attrs: Attrs = Field(default_factory=lambda: {})

    @classmethod
    def _is_supported_type(cls, type_):
        return type_ == cls or (
            typing.get_origin(type_) is Annotated and type_.__origin__ is cls
        )

    @abstractmethod
    def _handle_data_dump(self, data: np.ndarray) -> np.ndarray:
        """Hook into [Datastore.model_dump_hdf5][oqd_dataschema.datastore.Datastore.model_dump_hdf5] for compatibility mapping to HDF5."""
        pass

    @abstractmethod
    def _handle_data_load(self, data: np.ndarray) -> np.ndarray:
        """Hook into [Datastore.model_validate_hdf5][oqd_dataschema.datastore.Datastore.model_validate_hdf5] for reversing compatibility mapping, i.e. mapping data back to original type."""
        pass


# %%
