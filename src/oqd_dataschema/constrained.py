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


from typing import Annotated, Sequence, TypeAlias

from pydantic import AfterValidator

from oqd_dataschema.dataset import CastDataset
from oqd_dataschema.folder import Folder
from oqd_dataschema.table import CastTable
from oqd_dataschema.utils import _flex_shape_equal, _validator_from_condition

########################################################################################

__all__ = ["contable", "condataset", "confolder"]

########################################################################################


@_validator_from_condition
def _constrain_dim(model, *, min_dim=None, max_dim=None):
    """Constrains the dimension of a Dataset or Table."""

    if min_dim is not None and max_dim is not None and min_dim > max_dim:
        raise ValueError("Impossible to satisfy dimension constraints on dataset.")

    min_dim = 0 if min_dim is None else min_dim

    # fast escape
    if min_dim == 0 and max_dim is None:
        return

    dims = len(model.shape)
    if dims < min_dim or (max_dim is not None and dims > max_dim):
        raise ValueError(
            f"Expected {min_dim} <= dimension of shape{f' <= {max_dim}'}, but got shape = {model.shape}."
        )


@_validator_from_condition
def _constrain_shape(model, *, shape_constraint=None):
    """Constrains the shape of a Dataset or Table."""

    # fast escape
    if shape_constraint is None:
        return

    if not _flex_shape_equal(shape_constraint, model.shape):
        raise ValueError(
            f"Expected shape to be {shape_constraint}, but got {model.shape}."
        )


########################################################################################


@_validator_from_condition
def _constrain_dtype_dataset(dataset, *, dtype_constraint=None):
    """Constrains the dtype of a Dataset."""

    # fast escape
    if dtype_constraint is None:
        return

    # convert dtype constraint to set
    if (not isinstance(dtype_constraint, str)) and isinstance(
        dtype_constraint, Sequence
    ):
        dtype_constraint = set(dtype_constraint)
    elif isinstance(dtype_constraint, str):
        dtype_constraint = {dtype_constraint}

    # apply dtype constraint
    if dataset.dtype not in dtype_constraint:
        raise ValueError(
            f"Expected dtype to be of type one of {dtype_constraint}, but got {dataset.dtype}."
        )


def condataset(
    *, shape_constraint=None, dtype_constraint=None, min_dim=None, max_dim=None
) -> TypeAlias:
    """Implements dtype, dimension and shape constrains on the Dataset."""
    return Annotated[
        CastDataset,
        AfterValidator(_constrain_dtype_dataset(dtype_constraint=dtype_constraint)),
        AfterValidator(_constrain_dim(min_dim=min_dim, max_dim=max_dim)),
        AfterValidator(_constrain_shape(shape_constraint=shape_constraint)),
    ]


########################################################################################


@_validator_from_condition
def _constrain_dtype_table(table, *, dtype_constraint={}):
    """Constrains the dtype of a Table."""

    for k, v in dtype_constraint.items():
        if (not isinstance(v, str)) and isinstance(v, Sequence):
            _v = set(dtype_constraint[k])
        elif isinstance(v, str):
            _v = {dtype_constraint[k]}

        if _v and dict(table.columns)[k] not in _v:
            raise ValueError(
                f"Expected dtype to be of type one of {_v}, but got {dict(table.columns)[k]}."
            )


@_validator_from_condition
def _constrain_required_field(table, *, required_fields=None, strict_fields=False):
    """Constrains the fields of a Table."""

    if strict_fields and required_fields is None:
        raise ValueError("Constraints force an empty Table.")

    # fast escape
    if required_fields is None:
        return

    # convert required fields to set
    if (not isinstance(required_fields, str)) and isinstance(required_fields, Sequence):
        required_fields = set(required_fields)
    elif isinstance(required_fields, str):
        required_fields = {required_fields}

    diff = required_fields.difference(set([c[0] for c in table.columns]))
    reverse_diff = set([c[0] for c in table.columns]).difference(required_fields)

    if len(diff) > 0:
        raise ValueError(f"Missing required fields {diff}.")

    if strict_fields and len(reverse_diff):
        raise ValueError(
            f"Extra fields in the table are forbidden by constrains {reverse_diff}."
        )


def contable(
    *,
    required_fields=None,
    strict_fields=False,
    dtype_constraint={},
    shape_constraint=None,
    min_dim=None,
    max_dim=None,
) -> TypeAlias:
    """Implements field, dtype, dimension and shape constrains on the Table."""
    return Annotated[
        CastTable,
        AfterValidator(
            _constrain_required_field(
                required_fields=required_fields, strict_fields=strict_fields
            )
        ),
        AfterValidator(_constrain_dtype_table(dtype_constraint=dtype_constraint)),
        AfterValidator(_constrain_dim(min_dim=min_dim, max_dim=max_dim)),
        AfterValidator(_constrain_shape(shape_constraint=shape_constraint)),
    ]


########################################################################################


def confolder(
    *,
    shape_constraint=None,
    min_dim=None,
    max_dim=None,
) -> TypeAlias:
    """Implements dimension and shape constrains on the Folder."""
    return Annotated[
        Folder,
        AfterValidator(_constrain_dim(min_dim=min_dim, max_dim=max_dim)),
        AfterValidator(_constrain_shape(shape_constraint=shape_constraint)),
    ]
