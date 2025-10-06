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

########################################################################################

__all__ = ["_flex_shape_equal", "_validator_from_condition"]


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


def _validator_from_condition(f):
    """Helper decorator for turning a condition into a validation."""

    def _wrapped_validator(*args, **kwargs):
        def _wrapped_condition(model):
            f(model, *args, **kwargs)
            return model

        return _wrapped_condition

    return _wrapped_validator
