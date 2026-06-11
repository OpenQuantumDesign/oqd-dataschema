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

import json
import uuid

import numpy as np

from oqd_dataschema import AnalogEmulatorDataGroup, Datastore, Dataset, GroupRegistry


def test_analog_emulator_group_registered():
    assert "AnalogEmulatorDataGroup" in GroupRegistry.groups


def test_analog_emulator_hdf5_roundtrip(tmp_path):
    filepath = tmp_path / f"analog_{uuid.uuid4()}.h5"
    group = AnalogEmulatorDataGroup(
        attrs={"backend": "qutip", "dt": 0.001, "fock_cutoff": 4},
        times=Dataset(data=np.linspace(0, 1, 5, dtype=np.float64)),
        metrics=Dataset(
            data=np.zeros((5, 1), dtype=np.float64),
            attrs={"metric_labels": json.dumps(["Z"])},
        ),
        state=Dataset(data=np.zeros((5, 2), dtype=np.complex128)),
    )
    datastore = Datastore(groups={"emulation": group})
    datastore.model_dump_hdf5(filepath)
    reloaded = Datastore.model_validate_hdf5(filepath)
    sim = reloaded.groups["emulation"]
    assert sim.attrs["backend"] == "qutip"
    assert json.loads(sim.metrics.attrs["metric_labels"]) == ["Z"]
