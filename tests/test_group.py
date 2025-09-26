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

import uuid
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple

import numpy as np
import pytest

from oqd_dataschema import CastDataset, Dataset, GroupBase, condataset

########################################################################################


class TestGroupDefinition:
    @pytest.mark.parametrize(
        "field_type",
        [
            Dataset,
            CastDataset,
            Dict[str, Dataset],
            Dict[str, CastDataset],
            condataset(dtype_constraint="float32"),
            condataset(dtype_constraint=("float16", "float32", "float64")),
            condataset(min_dim=1),
            condataset(max_dim=1),
            condataset(min_dim=1, max_dim=2),
            condataset(shape_constraint=(1,)),
            condataset(shape_constraint=(None,)),
            condataset(shape_constraint=(None, 1)),
            condataset(shape_constraint=(None, None)),
            Optional[Dataset],
        ],
    )
    def test_data_field_definition(self, field_type):
        type(
            f"_Group_{uuid.uuid4()}".replace("-", ""),
            (GroupBase,),
            {"__annotations__": {"x": field_type}},
        )

    @pytest.mark.xfail(raises=TypeError)
    @pytest.mark.parametrize(
        "field_type",
        [
            Any,
            int,
            List[int],
            Tuple[int],
            List[Dataset],
            Tuple[Dataset],
            Dict[int, Dataset],
        ],
    )
    def test_invalid_data_field_definition(self, field_type):
        type(
            f"_Group_{uuid.uuid4()}".replace("-", ""),
            (GroupBase,),
            {"__annotations__": {"x": field_type}},
        )

    @pytest.mark.xfail(raises=AttributeError)
    def test_overwriting_attrs(self):
        type(
            f"_Group_{uuid.uuid4()}".replace("-", ""),
            (GroupBase,),
            {"__annotations__": {"attrs": Dict[str, Any]}},
        )

    @pytest.mark.xfail(raises=AttributeError)
    def test_overwriting_class_(self):
        groupname = f"_Group_{uuid.uuid4()}".replace("-", "")
        type(
            groupname,
            (GroupBase,),
            {"__annotations__": {"class_": Literal[groupname]}},
        )

    @pytest.mark.parametrize(
        ("field_type", "data"),
        [
            (Dataset, Dataset(data=np.random.rand(100))),
            (CastDataset, Dataset(data=np.random.rand(100))),
            (
                Dict[str, Dataset],
                {
                    "1": Dataset(data=np.random.rand(100)),
                    "2": Dataset(data=np.random.rand(100)),
                },
            ),
            (
                Dict[str, CastDataset],
                {
                    "1": Dataset(data=np.random.rand(100)),
                    "2": Dataset(data=np.random.rand(100)),
                },
            ),
            (condataset(dtype_constraint="float64"), Dataset(data=np.random.rand(100))),
            (
                condataset(dtype_constraint=("float16", "float32", "float64")),
                Dataset(data=np.random.rand(100)),
            ),
            (Optional[Dataset], Dataset(data=np.random.rand(100))),
            (Optional[Dataset], None),
        ],
    )
    def test_group_instantiation(self, field_type, data):
        _Group = type(
            f"_Group_{uuid.uuid4()}".replace("-", ""),
            (GroupBase,),
            {"__annotations__": {"x": field_type}},
        )

        _Group(x=data)

    @pytest.mark.parametrize(
        ("classvar_type"),
        [
            ClassVar,
            ClassVar[int],
        ],
    )
    def test_class_variable(self, classvar_type):
        type(
            f"_Group_{uuid.uuid4()}".replace("-", ""),
            (GroupBase,),
            {"__annotations__": {"x": classvar_type}},
        )

    @pytest.mark.parametrize(
        ("dataset"),
        [
            Dataset(),
            Dataset(data=np.random.rand(10)),
            Dataset(dtype="float64", shape=(10,)),
            Dataset(dtype="float64", shape=(10,), data=np.random.rand(10)),
        ],
    )
    def test_default_dataset(self, dataset):
        _Group = type(
            f"_Group_{uuid.uuid4()}".replace("-", ""),
            (GroupBase,),
            {"__annotations__": {"x": Dataset}, "x": dataset},
        )

        g = _Group()

        assert (
            (
                (g.x.data == dataset.data).all()
                and g.x.dtype == dataset.dtype
                and g.x.shape == dataset.shape
                and g.x.attrs == dataset.attrs
            )
            if isinstance(dataset.data, np.ndarray)
            else g.x == dataset
        )
