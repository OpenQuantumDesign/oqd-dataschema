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
from typing import Any, Dict, Literal, Optional

import h5py
import numpy as np
from pydantic import BaseModel, model_validator
from pydantic.types import TypeVar

from oqd_dataschema.base import Dataset, Group, GroupRegistry


# %%
class Datastore(BaseModel):
    """
    Saves the model and its associated data to an HDF5 file.
    This method serializes the model's data and attributes into an HDF5 file
    at the specified filepath.

    Attributes:
        filepath (pathlib.Path): The path to the HDF5 file where the model data will be saved.
    """

    groups: Dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def validate_groups(cls, data):
        if isinstance(data, dict) and "groups" in data:
            # Get the current adapter from registry
            try:
                adapter = GroupRegistry.get_adapter()
                validated_groups = {}

                for key, group_data in data["groups"].items():
                    if isinstance(group_data, Group):
                        # Already a Group instance
                        validated_groups[key] = group_data
                    elif isinstance(group_data, dict):
                        # Parse dict using discriminated union
                        validated_groups[key] = adapter.validate_python(group_data)
                    else:
                        raise ValueError(
                            f"Invalid group data for key '{key}': {type(group_data)}"
                        )

                data["groups"] = validated_groups

            except ValueError as e:
                if "No group types registered" in str(e):
                    raise ValueError(
                        "No group types available. Register group types before creating Datastore."
                    )
                raise

        return data

    def model_dump_hdf5(self, filepath: pathlib.Path, mode: Literal["w", "a"] = "a"):
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
            f.attrs["model"] = self.model_dump_json()

            # store each group
            for gkey, group in self.groups.items():
                if gkey in f.keys():
                    del f[gkey]
                h5_group = f.create_group(gkey)
                for akey, attr in group.attrs.items():
                    h5_group.attrs[akey] = attr

                for dkey, dataset in group.__dict__.items():
                    if not isinstance(dataset, Dataset):
                        continue
                    h5_dataset = h5_group.create_dataset(dkey, data=dataset.data)
                    for akey, attr in dataset.attrs.items():
                        h5_dataset.attrs[akey] = attr

    @classmethod
    def model_validate_hdf5(
        cls, filepath: pathlib.Path, types: Optional[TypeVar] = None
    ):
        """
        Loads the model from an HDF5 file at the specified filepath.

        Args:
            filepath (pathlib.Path): The path to the HDF5 file where the model data will be read and validated from.
        """
        with h5py.File(filepath, "r") as f:
            self = cls.model_validate_json(f.attrs["model"])

            # loop through all groups in the model schema and load HDF5 store
            for gkey, group in self.groups.items():
                for dkey, val in group.__dict__.items():
                    if dkey in ("attrs", "class_"):
                        continue
                    group.__dict__[dkey].data = np.array(f[gkey][dkey][()])
            return self
