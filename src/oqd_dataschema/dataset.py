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
from typing import Annotated, Any, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from oqd_dataschema.base import Attrs, DTypes

from .utils import _flex_shape_equal, _validator_from_condition

########################################################################################

__all__ = [
    "Dataset",
    "CastDataset",
    "condataset",
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
    """Implements dtype, dimension and shape constrains on the Dataset."""
    return Annotated[
        CastDataset,
        AfterValidator(_constrain_dtype(dtype_constraint=dtype_constraint)),
        AfterValidator(_constraint_dim(min_dim=min_dim, max_dim=max_dim)),
        AfterValidator(_constraint_shape(shape_constraint=shape_constraint)),
    ]
