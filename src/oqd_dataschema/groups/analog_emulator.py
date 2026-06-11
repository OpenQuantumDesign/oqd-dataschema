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

from typing import Optional

from oqd_dataschema.dataset import Dataset
from oqd_dataschema.group import GroupBase

########################################################################################

__all__ = ["AnalogEmulatorDataGroup"]

########################################################################################


class AnalogEmulatorDataGroup(GroupBase):
    """
    Processed output from an OQD analog classical emulator (e.g. QuTiP).

    Datasets:
        times: 1D float array of length ``n_tsteps``.
        metrics: 2D float array of shape ``(n_tsteps, n_metrics)`` with metric
            names stored in ``metrics.attrs["metric_labels"]`` (JSON list).
        state: 2D complex array of shape ``(n_tsteps, state_dim)`` for the state
            trajectory over the simulation.
        measurements: 2D int array of shape ``(n_shots, n_qubits)`` when shots
            were sampled after a measurement instruction.

    Group ``attrs`` hold run metadata (``dt``, ``n_shots``, ``fock_cutoff``,
    backend identifier, version, etc.).
    """

    times: Dataset
    metrics: Optional[Dataset] = None
    state: Optional[Dataset] = None
    measurements: Optional[Dataset] = None
