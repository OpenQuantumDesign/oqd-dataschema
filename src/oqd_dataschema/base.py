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
from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, Optional, Union

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
)

########################################################################################

__all__ = ["Attrs", "DTypes", "GroupField"]

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


class GroupField(BaseModel, ABC):
    attrs: Attrs

    @classmethod
    def _is_supported_type(cls, type_):
        return type_ == cls or (
            typing.get_origin(type_) is Annotated and type_.__origin__ is cls
        )

    @abstractmethod
    def _handle_data_dump(self, data):
        pass

    @abstractmethod
    def _handle_data_load(self, data):
        pass
