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

import numpy as np
import pytest

from oqd_dataschema import Dataset, Datastore, TrICalEmulatorDataGroup

########################################################################################


class TestTrICalEmulatorDataGroup:
    def test_serialize_deserialize_ket_trajectory(self, tmp_path):
        f = tmp_path / "trical_ket.h5"
        tspan = np.linspace(0.0, 1.0, 3)
        states = np.array(
            [[1.0, 0.0], [0.5, 0.5j], [0.0, 1.0]], dtype=np.complex128
        )
        final_state = states[-1]

        datastore = Datastore(
            groups={
                "emulation": TrICalEmulatorDataGroup(
                    tspan=Dataset(data=tspan),
                    states=Dataset(data=states),
                    final_state=Dataset(data=final_state),
                    attrs={
                        "backend": "qutip",
                        "solver": "SESolver",
                        "timestep": 0.5,
                        "hilbert_space": '{"E0": 2}',
                        "frame": "none",
                    },
                )
            }
        )

        datastore.model_dump_hdf5(f)
        loaded = Datastore.model_validate_hdf5(f)

        group = loaded["emulation"]
        np.testing.assert_allclose(group.tspan.data, tspan)
        np.testing.assert_allclose(group.states.data, states)
        np.testing.assert_allclose(group.final_state.data, final_state)
        assert group.attrs["backend"] == "qutip"
        assert group.attrs["solver"] == "SESolver"

    def test_serialize_deserialize_density_matrix_trajectory(self, tmp_path):
        f = tmp_path / "trical_density_matrix.h5"
        tspan = np.linspace(0.0, 1.0, 2)
        states = np.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.5, 0.0], [0.0, 0.5]],
            ],
            dtype=np.complex128,
        )
        final_state = states[-1]

        datastore = Datastore(
            groups={
                "emulation": TrICalEmulatorDataGroup(
                    tspan=Dataset(data=tspan),
                    states=Dataset(data=states),
                    final_state=Dataset(data=final_state),
                    attrs={"backend": "qutip", "solver": "MESolver"},
                )
            }
        )

        datastore.model_dump_hdf5(f)
        loaded = Datastore.model_validate_hdf5(f)

        group = loaded["emulation"]
        np.testing.assert_allclose(group.states.data, states)
        np.testing.assert_allclose(group.final_state.data, final_state)
        assert group.attrs["solver"] == "MESolver"

    @pytest.mark.xfail(raises=ValueError)
    def test_rejects_non_complex_states(self):
        TrICalEmulatorDataGroup(
            tspan=Dataset(data=np.linspace(0.0, 1.0, 2)),
            states=Dataset(data=np.ones((2, 2))),
            final_state=Dataset(data=np.ones(2, dtype=np.complex128)),
        )
