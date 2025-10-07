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
from typing import Annotated, Any, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from oqd_dataschema.base import Attrs, DTypes, GroupField

from .utils import _flex_shape_equal

########################################################################################

__all__ = [
    "Dataset",
    "CastDataset",
]

########################################################################################


class Dataset(GroupField, extra="forbid"):
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

    def _handle_data_dump(self, data):
        np_dtype = (
            np.dtypes.BytesDType if type(data.dtype) is np.dtypes.StrDType else None
        )

        if np_dtype is None:
            return data

        return data.astype(np_dtype)

    def _handle_data_load(self, data):
        np_dtype = DTypes.get(self.dtype).value
        return data.astype(np_dtype)


CastDataset = Annotated[Dataset, BeforeValidator(Dataset.cast)]
