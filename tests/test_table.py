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

import numpy as np
import pytest

from oqd_dataschema import Table
from oqd_dataschema.base import DTypes

########################################################################################


class TestTable:
    def test_empty_table(self):
        Table(columns=[], shape=(100,))

    @pytest.mark.xfail(raises=ValueError)
    @pytest.mark.parametrize(
        ("column"),
        [
            [("c1", "bool"), ("c1", "int16")],
            [
                ("c1", "str"),
                ("c2", "int16"),
                ("c3", "float64"),
                ("c1", "complex128"),
            ],
        ],
    )
    def test_duplicate_column(self, column):
        Table(columns=column, shape=(100,))


class TestTableDType:
    @pytest.mark.parametrize(
        ("dtype", "np_dtype"),
        [
            ("bool", np.dtypes.BoolDType()),
            ("int16", np.dtypes.Int16DType()),
            ("int32", np.dtypes.Int32DType()),
            ("int64", np.dtypes.Int64DType()),
            ("uint16", np.dtypes.UInt16DType()),
            ("uint32", np.dtypes.UInt32DType()),
            ("uint64", np.dtypes.UInt64DType()),
            ("float16", np.dtypes.Float16DType()),
            ("float32", np.dtypes.Float32DType()),
            ("float64", np.dtypes.Float64DType()),
            ("complex64", np.dtypes.Complex64DType()),
            ("complex128", np.dtypes.Complex128DType()),
            ("str", np.dtypes.StrDType(16)),
            ("bytes", np.dtypes.BytesDType(16)),
        ],
    )
    def test_dtypes(self, dtype, np_dtype):
        tbl = Table(columns=[("c", dtype)], shape=(100,))

        data = np.rec.fromarrays(
            np.random.rand(1, 100),
            dtype=np.dtype(
                [
                    ("c", np_dtype),
                ]
            ),
        )
        tbl.data = data

    @pytest.mark.parametrize(
        ("column", "np_dtype"),
        [
            (
                [("c1", "bool"), ("c2", "int16")],
                np.dtype(
                    [("c1", np.dtypes.BoolDType()), ("c2", np.dtypes.Int16DType())]
                ),
            ),
            (
                [
                    ("c1", "str"),
                    ("c2", "int16"),
                    ("c3", "float64"),
                    ("c4", "complex128"),
                ],
                np.dtype(
                    [
                        ("c1", np.dtypes.StrDType(16)),
                        ("c2", np.dtypes.Int16DType()),
                        ("c3", np.dtypes.Float64DType()),
                        ("c4", np.dtypes.Complex128DType()),
                    ]
                ),
            ),
        ],
    )
    def test_multi_column_dtypes(self, column, np_dtype):
        tbl = Table(columns=column, shape=(100,))

        data = np.rec.fromarrays(np.random.rand(len(column), 100), dtype=np_dtype)
        tbl.data = data

    @pytest.mark.xfail(raises=ValueError)
    @pytest.mark.parametrize(
        "dtype",
        [
            "bool",
            "int16",
            "int32",
            "int64",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
            "complex64",
            "complex128",
            "str",
            "bytes",
        ],
    )
    def test_unmatched_dtype_data(self, dtype):
        tbl = Table(columns=[("c", dtype)], shape=(100,))

        data = np.rec.fromarrays(
            np.random.rand(1, 100),
            dtype=np.dtype(
                [
                    ("c", "O"),
                ]
            ),
        )
        tbl.data = data

    @pytest.mark.parametrize(
        "np_dtype",
        [
            np.dtypes.BoolDType(),
            np.dtypes.Int16DType(),
            np.dtypes.Int32DType(),
            np.dtypes.Int64DType(),
            np.dtypes.UInt16DType(),
            np.dtypes.UInt32DType(),
            np.dtypes.UInt64DType(),
            np.dtypes.Float16DType(),
            np.dtypes.Float32DType(),
            np.dtypes.Float64DType(),
            np.dtypes.Complex64DType(),
            np.dtypes.Complex128DType(),
            np.dtypes.StrDType(16),
            np.dtypes.BytesDType(16),
        ],
    )
    def test_flexible_dtype(self, np_dtype):
        tbl = Table(columns=[("c", None)], shape=(100,))

        data = np.rec.fromarrays(
            np.random.rand(1, 100),
            dtype=np.dtype(
                [
                    ("c", np_dtype),
                ]
            ),
        )
        tbl.data = data

        assert (
            dict(tbl.columns)["c"]
            == DTypes(type(tbl.data.dtype.fields["c"][0])).name.lower()
        )

    def test_dtype_mutation(self):
        tbl = Table(columns=[("c", "float32")], shape=(100,))

        tbl.columns[0] = ("c", "float64")

        data = np.rec.fromarrays(
            np.random.rand(1, 100),
            dtype=np.dtype(
                [
                    ("c", "float64"),
                ]
            ),
        )
        tbl.data = data


# class TestDatasetShape:
#     @pytest.mark.xfail(raises=ValueError)
#     @pytest.mark.parametrize(
#         ("shape", "data_shape"),
#         [
#             ((0,), (100,)),
#             ((1,), (100,)),
#             ((99,), (100,)),
#             ((1, 1), (100,)),
#             ((100, None), (100,)),
#             ((None, None), (100,)),
#             ((None, 100), (100,)),
#         ],
#     )
#     def test_unmatched_shape_data(self, shape, data_shape):
#         ds = Dataset(dtype="float64", shape=shape)

#         data = np.random.rand(*data_shape)
#         ds.data = data

#     @pytest.mark.parametrize(
#         ("shape", "data_shape"),
#         [
#             ((None,), (0,)),
#             ((None,), (1,)),
#             ((None,), (100,)),
#             ((None, 0), (0, 0)),
#             ((None, 1), (1, 1)),
#             ((None, None), (1, 1)),
#             ((None, None), (10, 100)),
#             ((None, None, 1), (1, 1, 1)),
#         ],
#     )
#     def test_flexible_shape(self, shape, data_shape):
#         ds = Dataset(dtype="float64", shape=shape)

#         data = np.random.rand(*data_shape)
#         ds.data = data

#         assert ds.shape == ds.data.shape

#     def test_shape_mutation(self):
#         ds = Dataset(dtype="float64", shape=(1,))

#         ds.shape = (100,)

#         data = np.random.rand(100)
#         ds.data = data


# class TestCastDataset:
#     @pytest.fixture
#     def adapter(self):
#         return TypeAdapter(CastDataset)

#     @pytest.mark.parametrize(
#         ("data", "dtype", "shape"),
#         [
#             (np.random.rand(100), "float64", (100,)),
#             (np.random.rand(10).astype("str"), "str", (10,)),
#             (np.random.rand(1, 10, 100).astype("bytes"), "bytes", (1, 10, 100)),
#         ],
#     )
#     def test_cast(self, adapter, data, shape, dtype):
#         ds = adapter.validate_python(data)

#         assert ds.shape == shape and ds.dtype == dtype


# class TestConstrainedDataset:
#     @pytest.mark.parametrize(
#         ("cds", "data"),
#         [
#             (condataset(dtype_constraint="float64"), np.random.rand(10)),
#             (condataset(dtype_constraint="str"), np.random.rand(10).astype(str)),
#             (
#                 condataset(dtype_constraint=("float16", "float32", "float64")),
#                 np.random.rand(10),
#             ),
#             (
#                 condataset(dtype_constraint=("float16", "float32", "float64")),
#                 np.random.rand(10).astype("float16"),
#             ),
#             (
#                 condataset(dtype_constraint=("float16", "float32", "float64")),
#                 np.random.rand(10).astype("float32"),
#             ),
#         ],
#     )
#     def test_constrained_dataset_dtype(self, cds, data):
#         adapter = TypeAdapter(cds)

#         adapter.validate_python(data)

#     @pytest.mark.xfail(raises=ValueError)
#     @pytest.mark.parametrize(
#         ("cds", "data"),
#         [
#             (condataset(dtype_constraint="float64"), np.random.rand(10).astype(str)),
#             (condataset(dtype_constraint="str"), np.random.rand(10)),
#             (
#                 condataset(dtype_constraint=("float16", "float32", "float64")),
#                 np.random.rand(10).astype(str),
#             ),
#         ],
#     )
#     def test_violate_dtype_constraint(self, cds, data):
#         adapter = TypeAdapter(cds)

#         adapter.validate_python(data)

#     @pytest.mark.parametrize(
#         ("cds", "data"),
#         [
#             (condataset(min_dim=1, max_dim=1), np.random.rand(10)),
#             (condataset(min_dim=0, max_dim=1), np.random.rand(10)),
#             (condataset(max_dim=2), np.random.rand(10)),
#             (condataset(max_dim=3), np.random.rand(10, 10, 10)),
#             (condataset(min_dim=2), np.random.rand(10, 10)),
#             (condataset(min_dim=2), np.random.rand(10, 10, 10, 10, 10)),
#             (condataset(min_dim=2, max_dim=4), np.random.rand(10, 10, 10, 10)),
#             (condataset(min_dim=2, max_dim=4), np.random.rand(10, 10, 10)),
#             (condataset(min_dim=2, max_dim=4), np.random.rand(10, 10)),
#         ],
#     )
#     def test_constrained_dataset_dimension(self, cds, data):
#         adapter = TypeAdapter(cds)

#         adapter.validate_python(data)

#     @pytest.mark.xfail(raises=ValueError)
#     @pytest.mark.parametrize(
#         ("cds", "data"),
#         [
#             (condataset(min_dim=1, max_dim=1), np.random.rand(10, 10)),
#             (condataset(min_dim=2, max_dim=3), np.random.rand(10)),
#             (condataset(min_dim=2, max_dim=3), np.random.rand(10, 10, 10, 10)),
#         ],
#     )
#     def test_violate_dimension_constraint(self, cds, data):
#         adapter = TypeAdapter(cds)

#         adapter.validate_python(data)

#     @pytest.mark.parametrize(
#         ("cds", "data"),
#         [
#             (condataset(shape_constraint=(None,)), np.random.rand(10)),
#             (condataset(shape_constraint=(10,)), np.random.rand(10)),
#             (condataset(shape_constraint=(None, None)), np.random.rand(1, 2)),
#             (condataset(shape_constraint=(1, None)), np.random.rand(1, 2)),
#             (condataset(shape_constraint=(1, 2)), np.random.rand(1, 2)),
#             (condataset(shape_constraint=(1, None, 3)), np.random.rand(1, 10, 3)),
#         ],
#     )
#     def test_constrained_dataset_shape(self, cds, data):
#         adapter = TypeAdapter(cds)

#         adapter.validate_python(data)

#     @pytest.mark.xfail(raises=ValueError)
#     @pytest.mark.parametrize(
#         ("cds", "data"),
#         [
#             (condataset(shape_constraint=(1,)), np.random.rand(10)),
#             (condataset(shape_constraint=(None,)), np.random.rand(10, 10)),
#             (condataset(shape_constraint=(None, 1)), np.random.rand(10, 10)),
#             (condataset(shape_constraint=(None, 1)), np.random.rand(1, 10)),
#         ],
#     )
#     def test_violate_shape_constraint(self, cds, data):
#         adapter = TypeAdapter(cds)

#         adapter.validate_python(data)
