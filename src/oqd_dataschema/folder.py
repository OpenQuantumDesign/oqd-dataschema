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


from types import MappingProxyType
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import (
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import TypeAliasType

from oqd_dataschema.base import Attrs, DTypes, GroupField
from oqd_dataschema.utils import _flex_shape_equal

########################################################################################

__all__ = [
    "Folder",
]

########################################################################################

DocumentSchema = TypeAliasType(
    "DocumentSchema",
    Dict[str, Union["DocumentSchema", Optional[Literal[DTypes.names()]]]],  # type: ignore
)


class Folder(GroupField, extra="forbid"):
    document_schema: DocumentSchema
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

        if not isinstance(value.dtype.fields, MappingProxyType):
            raise TypeError("dtype of data must be a structured dtype.")

        value = value.view(np.recarray)

        return value

    @staticmethod
    def _is_valid_array(document_schema, data_dtype, position=""):
        # check if data_dtype is a structured dtype
        if not isinstance(data_dtype.fields, MappingProxyType):
            raise TypeError(
                f"Error {f'in key `{position}`' if position else 'at root'}, expected structured dtype matching {document_schema = } but got unstructured dtype {data_dtype = }."
            )

        # check if fields all match
        if set(document_schema.keys()) != set(data_dtype.fields.keys()):
            diff = set(document_schema.keys()).difference(set(data_dtype.fields.keys()))
            rv_diff = set(data_dtype.fields.keys()).difference(
                set(document_schema.keys())
            )
            raise ValueError(
                f"Error {f'in key `{position}`' if position else 'at root '}, mismatched {'subkeys' if position else 'keys'} between `document_schema` (unmatched = {diff}) and numpy data structured dtype (unmatched = {rv_diff})."
            )

        # recursively check document_schema matches structured dtype data_dtype
        for k, v in document_schema.items():
            if isinstance(v, dict):
                Folder._is_valid_array(
                    v, data_dtype.fields[k][0], position + "." + k if position else k
                )
                continue

            # check if dtypes match
            if (
                v is not None
                and type(data_dtype.fields[k][0]) is not DTypes.get(v).value
            ):
                raise ValueError(
                    f"Error {f'in key `{position}`' if position else 'at root '}, expected {'subkey' if position else 'key'} `{k}` to be of dtype compatible with {v} but got dtype {data_dtype.fields[k][0]}."
                )

    @model_validator(mode="after")
    def validate_data_matches_shape_dtype(self):
        """Ensure that `data` matches `dtype` and `shape`."""

        # check if data exist
        if self.data is None:
            return self

        # check if document_schema matches the data's structured dtype
        self._is_valid_array(self.document_schema, self.data.dtype)

        # check if shape mataches data
        if self.shape is not None and not _flex_shape_equal(
            self.data.shape, self.shape
        ):
            raise ValueError(f"Expected shape {self.shape}, but got {self.data.shape}.")

        # resassign shape to concrete value if it is None or a flexible shape
        if self.shape != self.data.shape:
            self.shape = self.data.shape

        return self

    @staticmethod
    def _dump_dtype_str_to_bytes(dtype):
        np_dtype = []

        for k, (v, _) in dtype.fields.items():
            if isinstance(v.fields, MappingProxyType):
                dt = Folder._dump_dtype_str_to_bytes(v)
            elif type(v) is np.dtypes.StrDType:
                dt = np.empty(0, dtype=v).astype(np.dtypes.BytesDType).dtype
            else:
                dt = v

            np_dtype.append((k, dt))

        return np.dtype(np_dtype)

    def _handle_data_dump(self, data):
        np_dtype = self._dump_dtype_str_to_bytes(data.dtype)

        return data.astype(np_dtype)

    @staticmethod
    def _load_dtype_bytes_to_str(document_schema, dtype):
        np_dtype = []

        for k, (v, _) in dtype.fields.items():
            if isinstance(v.fields, MappingProxyType):
                dt = Folder._load_dtype_bytes_to_str(document_schema[k], v)
            elif document_schema[k] == "str":
                dt = np.empty(0, dtype=v).astype(np.dtypes.StrDType).dtype
            else:
                dt = v

            np_dtype.append((k, dt))

        return np.dtype(np_dtype)

    def _handle_data_load(self, data):
        np_dtype = self._load_dtype_bytes_to_str(self.document_schema, data.dtype)

        return data.astype(np_dtype)
