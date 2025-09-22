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
import pathlib

import numpy as np

from oqd_dataschema.base import Dataset, GroupBase
from oqd_dataschema.datastore import Datastore
from oqd_dataschema.groups import (
    SinaraRawDataGroup,
)


# %%
def test_adapt():
    class TestNewGroup(GroupBase):
        """ """

        array: Dataset

    filepath = pathlib.Path("test.h5")

    data = np.ones([10, 10]).astype("int64")
    group1 = TestNewGroup(array=Dataset(data=data))

    data = np.ones([10, 10]).astype("int32")
    group2 = SinaraRawDataGroup(camera_images=Dataset(data=data))

    datastore = Datastore(
        groups={
            "group1": group1,
            "group2": group2,
        }
    )
    datastore.model_dump_hdf5(filepath, mode="w")

    Datastore.model_validate_hdf5(filepath)
