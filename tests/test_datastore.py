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

import pytest 
#%%
import pathlib

import numpy as np

from oqd_dataschema.base import Dataset, mapping
from oqd_dataschema.datastore import Datastore
from oqd_dataschema.groups import (
    SinaraRawDataGroup,
)

#%%         
@pytest.mark.parametrize(
    'dtype', [
        "int32",
        "int64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ]
)
def test_serialize_deserialize(dtype):
    data = np.ones([10, 10]).astype(dtype)
    dataset = SinaraRawDataGroup(camera_images=Dataset(data=data))
    data = Datastore(groups={"test": dataset})
    
    filepath = pathlib.Path("test.h5")
    data.model_dump_hdf5(filepath)
    
    data_reload = Datastore.model_validate_hdf5(filepath)

    assert data_reload.groups['test'].camera_images.data.dtype == mapping[dtype]
    
