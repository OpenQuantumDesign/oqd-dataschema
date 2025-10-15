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

import pytest

from oqd_dataschema import (
    CastDataset,
    Dataset,
    GroupBase,
    GroupRegistry,
    condataset,
)


class TestGroupRegistry:
    def test_clear(self):
        GroupRegistry.clear()

        GroupRegistry.groups = dict()

    def test_add_group(self):
        GroupRegistry.clear()

        groups = set()
        for k in "ABCDE":
            groups.add(
                type(f"_Group{k}", (GroupBase,), {"__annotations__": {"x": Dataset}})
            )

        assert set(GroupRegistry.groups.values()) == groups

    def test_overwrite_group(self):
        GroupRegistry.clear()

        _GroupA = type("_GroupA", (GroupBase,), {"__annotations__": {"x": Dataset}})

        assert set(GroupRegistry.groups.values()) == {_GroupA}

        with pytest.warns(UserWarning):
            _mGroupA = type(
                "_GroupA", (GroupBase,), {"__annotations__": {"x": CastDataset}}
            )

        assert set(GroupRegistry.groups.values()) == {_mGroupA}

    @pytest.fixture
    def group_generator(self):
        def _groupgen():
            groups = []
            for k, dtype in zip(
                "ABCDE",
                ["str", "float64", "bytes", "bool", ("int16", "int32", "int64")],
            ):
                groups.append(
                    type(
                        f"_Group{k}",
                        (GroupBase,),
                        {"__annotations__": {"x": condataset(dtype_constraint=dtype)}},
                    )
                )
            return groups

        return _groupgen
