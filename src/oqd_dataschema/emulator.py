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

from oqd_dataschema.constrained import condataset
from oqd_dataschema.group import GroupBase

########################################################################################

__all__ = ["TrICalEmulatorDataGroup"]

########################################################################################


class TrICalEmulatorDataGroup(GroupBase):
    """
    Schema for TrICal emulator time-evolution output.

    Attributes:
        tspan: 1D array of saved times.
        states: Complex state trajectory. Kets use shape `(n_tsteps, hilbert_dim)`;
            density matrices use shape `(n_tsteps, hilbert_dim, hilbert_dim)`.
        final_state: Complex final ket or density matrix.
    """

    tspan: condataset(dtype_constraint=("float32", "float64"), min_dim=1, max_dim=1)
    states: condataset(
        dtype_constraint=("complex64", "complex128"), min_dim=2, max_dim=3
    )
    final_state: condataset(
        dtype_constraint=("complex64", "complex128"), min_dim=1, max_dim=2
    )
