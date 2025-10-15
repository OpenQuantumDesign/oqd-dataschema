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

from functools import reduce
from types import MappingProxyType

import numpy as np
from numpy.lib import recfunctions as rfn

########################################################################################

__all__ = [
    "unstructured_to_structured",
    "dict_to_structured",
]


########################################################################################


def _unstructured_to_structured_helper(data, dtype):
    for n, (k, (v, _)) in enumerate(dtype.fields.items()):
        if isinstance(v.fields, MappingProxyType):
            x = _unstructured_to_structured_helper(data, v)

        else:
            x = data.pop(0).astype(type(v))

        if n == 0:
            new_data = x.astype(np.dtype([(k, x.dtype)]))
        else:
            if new_data.shape != x.shape:
                raise ValueError(
                    f"Incompatible shape, expected {new_data.shape} but got {x.shape}."
                )

            new_data = rfn.append_fields(
                new_data.flatten(), k, x.flatten(), usemask=False
            ).reshape(x.shape)

    return new_data.view(np.recarray)


def unstructured_to_structured(data, dtype):
    data = list(np.moveaxis(data, -1, 0))

    leaves = len(rfn.flatten_descr(dtype))
    if len(data) != leaves:
        raise ValueError(
            f"Incompatible shape, last dimension of data ({data.shape[-1]}) must match number of leaves in structured dtype ({leaves})."
        )

    new_data = _unstructured_to_structured_helper(data, dtype)

    return new_data


########################################################################################


def _dtype_from_dict(data):
    np_dtype = []

    for k, v in data.items():
        if isinstance(v, dict):
            dt = _dtype_from_dict(v)
        else:
            dt = v.dtype

        np_dtype.append((k, dt))

    return np.dtype(np_dtype)


def _dict_to_structured_helper(data, dtype):
    for n, (k, (v, _)) in enumerate(dtype.fields.items()):
        if isinstance(v.fields, MappingProxyType):
            x = _dict_to_structured_helper(data[k], v)
        else:
            x = data[k]

        if n == 0:
            new_data = x.astype(np.dtype([(k, x.dtype)]))
        else:
            if new_data.shape != x.shape:
                raise ValueError(
                    f"Incompatible shape, expected {new_data.shape} but got {x.shape}."
                )

            new_data = rfn.append_fields(
                new_data.flatten(), k, x.flatten(), usemask=False
            ).reshape(x.shape)

    return new_data.view(np.recarray)


def dict_to_structured(data):
    data_dtype = _dtype_from_dict(data)
    new_data = _dict_to_structured_helper(data, dtype=data_dtype)
    return new_data


########################################################################################


def _flex_shape_equal(shape1, shape2):
    """Helper function for comparing concrete and flex shapes."""
    return len(shape1) == len(shape2) and reduce(
        lambda x, y: x and y,
        map(
            lambda x: x[0] is None or x[1] is None or x[0] == x[1],
            zip(shape1, shape2),
        ),
    )


########################################################################################


def _validator_from_condition(f):
    """Helper decorator for turning a condition into a validation."""

    def _wrapped_validator(*args, **kwargs):
        def _wrapped_condition(model):
            f(model, *args, **kwargs)
            return model

        return _wrapped_condition

    return _wrapped_validator


########################################################################################


def _is_list_unique(data):
    seen = set()
    duplicates = set()
    for element in data:
        if element in duplicates:
            continue

        if element in seen:
            duplicates.add(element)
            continue

        seen.add(element)

    return (duplicates == set(), duplicates)
