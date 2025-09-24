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

import json
import pathlib
from typing import Any, Dict, Literal

import h5py
import numpy as np
from pydantic import (
    BaseModel,
    field_validator,
)

from oqd_dataschema.base import Attrs, Dataset, DTypes, GroupBase, GroupRegistry

########################################################################################

__all__ = ["Datastore"]

########################################################################################


# %%
class Datastore(BaseModel, extra="forbid"):
    """
    Saves the model and its associated data to an HDF5 file.
    This method serializes the model's data and attributes into an HDF5 file
    at the specified filepath.

    Attributes:
        filepath (pathlib.Path): The path to the HDF5 file where the model data will be saved.
    """

    groups: Dict[str, Any]

    attrs: Attrs = {}

    @field_validator("groups", mode="before")
    @classmethod
    def validate_groups(cls, data):
        if isinstance(data, dict):
            # Get the current adapter from registry
            try:
                validated_groups = {}

                for key, group_data in data.items():
                    if isinstance(group_data, GroupBase):
                        # Already a Group instance
                        validated_groups[key] = group_data
                    elif isinstance(group_data, dict):
                        # Parse dict using discriminated union
                        validated_groups[key] = GroupRegistry.adapter.validate_python(
                            group_data
                        )
                    else:
                        raise ValueError(
                            f"Invalid group data for key '{key}': {type(group_data)}"
                        )

                data = validated_groups

            except ValueError as e:
                if "No group types registered" in str(e):
                    raise ValueError(
                        "No group types available. Register group types before creating Datastore."
                    )
                raise

        return data

    def model_dump_hdf5(self, filepath: pathlib.Path, mode: Literal["w", "a"] = "w"):
        """
        Saves the model and its associated data to an HDF5 file.
        This method serializes the model's data and attributes into an HDF5 file
        at the specified filepath.

        Args:
            filepath (pathlib.Path): The path to the HDF5 file where the model data will be saved.
        """
        filepath.parent.mkdir(exist_ok=True, parents=True)

        with h5py.File(filepath, mode) as f:
            # store the model JSON schema
            f.attrs["_model_signature"] = self.model_dump_json()
            for akey, attr in self.attrs.items():
                f.attrs[akey] = attr

            # store each group
            for gkey, group in self.groups.items():
                if gkey in f.keys():
                    del f[gkey]
                h5_group = f.create_group(gkey)

                h5_group.attrs["_model_schema"] = json.dumps(group.model_json_schema())
                for akey, attr in group.attrs.items():
                    h5_group.attrs[akey] = attr

                for dkey, dataset in group.__dict__.items():
                    if not isinstance(dataset, Dataset):
                        continue

                    if dataset.dtype in "str":
                        h5_dataset = h5_group.create_dataset(
                            dkey, data=dataset.data.astype(np.dtypes.BytesDType)
                        )
                    else:
                        h5_dataset = h5_group.create_dataset(dkey, data=dataset.data)

                    for akey, attr in dataset.attrs.items():
                        h5_dataset.attrs[akey] = attr

    @classmethod
    def model_validate_hdf5(cls, filepath: pathlib.Path):
        """
        Loads the model from an HDF5 file at the specified filepath.

        Args:
            filepath (pathlib.Path): The path to the HDF5 file where the model data will be read and validated from.
        """
        with h5py.File(filepath, "r") as f:
            self = cls.model_validate_json(f.attrs["_model_signature"])

            # loop through all groups in the model schema and load HDF5 store
            for gkey, group in self:
                for dkey in group.__class__.model_fields:
                    if dkey in ("attrs", "class_"):
                        continue
                    group.__dict__[dkey].data = np.array(f[gkey][dkey][()]).astype(
                        DTypes.get(group.__dict__[dkey].dtype).value
                    )
            return self

    def __getitem__(self, key):
        return self.groups.__getitem__(key)

    def __iter__(self):
        return self.groups.items().__iter__()
