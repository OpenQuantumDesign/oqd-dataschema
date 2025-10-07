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


import typing
from types import MappingProxyType
from typing import Annotated, Any, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from oqd_dataschema.base import Attrs, DTypes
from oqd_dataschema.utils import (
    _flex_shape_equal,
    _is_list_unique,
)

########################################################################################

__all__ = [
    "Table",
    "CastTable",
]

########################################################################################


Column = Tuple[str, Optional[Literal[DTypes.names()]]]


class Table(BaseModel, extra="forbid"):
    columns: List[Column]  # type: ignore
    shape: Optional[Tuple[Union[int, None], ...]] = None
    data: Optional[Any] = Field(default=None, exclude=True)

    attrs: Attrs = {}

    model_config = ConfigDict(
        use_enum_values=False, arbitrary_types_allowed=True, validate_assignment=True
    )

    @field_validator("columns", mode="before")
    @classmethod
    def validate_unique(cls, value):
        column_names = [c[0] for c in value]

        is_unique, duplicates = _is_list_unique(column_names)
        if not is_unique:
            raise ValueError(f"More than one column with the same name ({duplicates}).")

        return value

    @property
    def dataframe(self):
        if len(self.shape) > 1:
            raise ValueError(
                "Conversion to pandas DataFrame only supported on 1D Table."
            )
        return pd.DataFrame(
            data=self.data, columns=[c[0] for c in self.columns]
        ).astype({k: v for k, v in self.columns})

    @staticmethod
    def _pd_to_np(df):
        np_dtype = []
        for k, v in df.dtypes.items():
            if type(v) is not np.dtypes.ObjectDType:
                field_np_dtype = (k, v)
                np_dtype.append(field_np_dtype)
                continue

            # Check if column of object dtype is actually str dtype
            if (np.vectorize(lambda x: isinstance(x, str))(df[k].to_numpy())).all():
                dt = df[k].to_numpy().astype(np.dtypes.StrDType).dtype
                field_np_dtype = (k, dt)

                np_dtype.append(field_np_dtype)
                continue

            raise ValueError(f"Unsupported datatype for column {k}")

        return np.rec.fromarrays(
            df.to_numpy().transpose(),
            names=[dt[0] for dt in np_dtype],
            formats=[dt[1] for dt in np_dtype],
        ).astype(np.dtype(np_dtype))

    @field_validator("data", mode="before")
    @classmethod
    def validate_and_update(cls, value):
        # check if data exist
        if value is None:
            return value

        # check if data is a numpy array
        if not isinstance(value, (np.ndarray, pd.DataFrame)):
            raise TypeError("`data` must be a numpy.ndarray or pandas.DataFrame.")

        if isinstance(value, pd.DataFrame):
            value = cls._pd_to_np(value)

        if not isinstance(value.dtype.fields, MappingProxyType):
            raise TypeError("dtype of data must be a structured dtype.")

        if isinstance(value, np.ndarray):
            value = value.view(np.recarray)

        return value

    @model_validator(mode="after")
    def validate_data_matches_shape_dtype(self):
        """Ensure that `data` matches `dtype` and `shape`."""

        # check if data exist
        if self.data is None:
            return self

        if set(self.data.dtype.fields.keys()) != set([c[0] for c in self.columns]):
            raise ValueError("Fields of data do not match expected field for Table.")

        # check if dtype matches data
        for k, v in self.data.dtype.fields.items():
            if (
                dict(self.columns)[k] is not None
                and type(v[0]) is not DTypes.get(dict(self.columns)[k]).value
            ):
                raise ValueError(
                    f"Expected data dtype `{dict(self.columns)[k]}`, but got `{v[0].name}`."
                )

        # check if shape mataches data
        if self.shape is not None and not _flex_shape_equal(
            self.data.shape, self.shape
        ):
            raise ValueError(f"Expected shape {self.shape}, but got {self.data.shape}.")

        # reassign dtype if it is None
        for n, (k, v) in enumerate(self.columns):
            if v != DTypes(type(self.data.dtype.fields[k][0])).name.lower():
                self.columns[n] = (
                    k,
                    DTypes(type(self.data.dtype.fields[k][0])).name.lower(),
                )

        # resassign shape to concrete value if it is None or a flexible shape
        if self.shape != self.data.shape:
            self.shape = self.data.shape

        return self

    @classmethod
    def cast(cls, data):
        if isinstance(data, pd.DataFrame):
            data = cls._pd_to_np(data)

        if isinstance(data, np.ndarray):
            if not isinstance(data.dtype.fields, MappingProxyType):
                raise TypeError("dtype of data must be a structured dtype.")

            columns = [
                (k, DTypes(type(v)).name.lower())
                for k, (v, _) in data.dtype.fields.items()
            ]

            return cls(columns=columns, data=data)
        return data

    @classmethod
    def _is_table_type(cls, type_):
        return type_ == cls or (
            typing.get_origin(type_) is Annotated and type_.__origin__ is cls
        )


CastTable = Annotated[Table, BeforeValidator(Table.cast)]
