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
from typing import Dict, Optional

import numpy as np
import pytest

from oqd_dataschema import Dataset, Datastore, GroupBase

# %%

_Group = type(
    f"_Group_{uuid.uuid4()}".replace("-", ""),
    (GroupBase,),
    {
        "__annotations__": {
            "x": Dataset,
            "y": Dict[str, Dataset],
            "z": Optional[Dataset],
        },
        "y": {},
        "z": None,
    },
)


class TestDatastore:
    @pytest.mark.parametrize(
        ("dtype", "np_dtype"),
        [
            ("bool", np.dtypes.BoolDType),
            ("int16", np.dtypes.Int16DType),
            ("int32", np.dtypes.Int32DType),
            ("int64", np.dtypes.Int64DType),
            ("uint16", np.dtypes.UInt16DType),
            ("uint32", np.dtypes.UInt32DType),
            ("uint64", np.dtypes.UInt64DType),
            ("float16", np.dtypes.Float16DType),
            ("float32", np.dtypes.Float32DType),
            ("float64", np.dtypes.Float64DType),
            ("complex64", np.dtypes.Complex64DType),
            ("complex128", np.dtypes.Complex128DType),
            ("str", np.dtypes.StrDType),
            ("bytes", np.dtypes.BytesDType),
            ("string", np.dtypes.StringDType),
        ],
    )
    def test_serialize_deserialize_dtypes(self, dtype, np_dtype, tmp_path):
        f = tmp_path / f"tmp{uuid.uuid4()}.h5"

        datastore = Datastore(
            groups={"g1": _Group(x=Dataset(data=np.random.rand(1).astype(np_dtype)))}
        )

        datastore.model_dump_hdf5(f)

        Datastore.model_validate_hdf5(f)

    @pytest.mark.parametrize(
        ("x", "y", "z"),
        [
            (
                Dataset(data=np.random.rand(10)),
                {},
                None,
            ),
            (
                Dataset(data=np.random.rand(10)),
                {"f1": Dataset(data=np.random.rand(10))},
                None,
            ),
            (
                Dataset(data=np.random.rand(10)),
                {"f1": Dataset(data=np.random.rand(10))},
                Dataset(data=np.random.rand(10)),
            ),
            (
                Dataset(data=np.random.rand(10)),
                {
                    "f1": Dataset(data=np.random.rand(10)),
                    "f2": Dataset(data=np.random.rand(10)),
                },
                Dataset(data=np.random.rand(10)),
            ),
        ],
    )
    def test_serialize_deserialize_dataset_types(self, x, y, z, tmp_path):
        f = tmp_path / f"tmp{uuid.uuid4()}.h5"

        datastore = Datastore(groups={"g1": _Group(x=x, y=y, z=z)})

        datastore.model_dump_hdf5(f)

        Datastore.model_validate_hdf5(f)
