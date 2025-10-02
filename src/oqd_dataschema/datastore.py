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
    Class representing a datastore with restricted HDF5 format.

    Attributes:
        groups (Dict[str,Group]): groups of data.
        attrs (Attrs): attributes of the datastore.
    """

    groups: Dict[str, Any]

    attrs: Attrs = {}

    @classmethod
    def _validate_group(cls, key, group):
        """Helper function for validating group to be of type Group registered in the GroupRegistry."""
        if isinstance(group, GroupBase):
            return group

        if isinstance(group, dict):
            return GroupRegistry.adapter.validate_python(group)

        raise ValueError(f"Key `{key}` contains invalid group data.")

    @field_validator("groups", mode="before")
    @classmethod
    def validate_groups(cls, data):
        """Validates groups to be of type Group registered in the GroupRegistry."""
        if GroupRegistry.groups == {}:
            raise ValueError(
                "No group types available. Register group types before creating Datastore."
            )

        validated_groups = {k: cls._validate_group(k, v) for k, v in data.items()}
        return validated_groups

    def _dump_group(self, h5datastore, gkey, group):
        """Helper function for dumping Group."""
        # remove existing group
        if gkey in h5datastore.keys():
            del h5datastore[gkey]

        # create group
        h5_group = h5datastore.create_group(gkey)

        # dump group schema
        h5_group.attrs["_group_schema"] = json.dumps(
            group.model_json_schema(), indent=2
        )

        # dump group attributes
        for akey, attr in group.attrs.items():
            h5_group.attrs[akey] = attr

        # dump group data
        for dkey, dataset in group.__dict__.items():
            if dkey in ["attr", "class_"]:
                continue

            # if group field contain dictionary of Dataset
            if isinstance(dataset, dict):
                h5_subgroup = h5_group.create_group(dkey)
                for ddkey, ddataset in dataset.items():
                    self._dump_dataset(h5_subgroup, ddkey, ddataset)
                continue

            self._dump_dataset(h5_group, dkey, dataset)

    def _dump_dataset(self, h5group, dkey, dataset):
        """Helper function for dumping Dataset."""
        if not isinstance(dataset, Dataset):
            raise ValueError("Group data field is not a Dataset.")

        # dtype str converted to bytes when dumped (h5 compatibility)
        if dataset.dtype in "str":
            h5_dataset = h5group.create_dataset(
                dkey, data=dataset.data.astype(np.dtypes.BytesDType)
            )
        else:
            h5_dataset = h5group.create_dataset(dkey, data=dataset.data)

        # dump dataset attributes
        for akey, attr in dataset.attrs.items():
            h5_dataset.attrs[akey] = attr

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
            # dump the datastore signature
            f.attrs["_datastore_signature"] = self.model_dump_json(indent=2)
            for akey, attr in self.attrs.items():
                f.attrs[akey] = attr

            # dump each group
            for gkey, group in self.groups.items():
                if gkey in ["attr", "class_"]:
                    continue

                self._dump_group(f, gkey, group)

    @classmethod
    def model_validate_hdf5(cls, filepath: pathlib.Path):
        """
        Loads the model from an HDF5 file at the specified filepath.

        Args:
            filepath (pathlib.Path): The path to the HDF5 file where the model data will be read and validated from.
        """
        with h5py.File(filepath, "r") as f:
            # Load datastore signature
            self = cls.model_validate_json(f.attrs["_datastore_signature"])

            # loop through all groups in the model schema and load the data
            for gkey, group in self:
                for dkey in group.__class__.model_fields:
                    # ignore attrs and class_ fields
                    if dkey in ("attrs", "class_"):
                        continue

                    # load Dataset data
                    if isinstance(group.__dict__[dkey], Dataset):
                        group.__dict__[dkey].data = np.array(f[gkey][dkey][()]).astype(
                            DTypes.get(group.__dict__[dkey].dtype).value
                        )
                        continue

                    # load data for dict of Dataset
                    if isinstance(group.__dict__[dkey], dict):
                        for ddkey in group.__dict__[dkey]:
                            group.__dict__[dkey][ddkey].data = np.array(
                                f[gkey][dkey][ddkey][()]
                            ).astype(
                                DTypes.get(group.__dict__[dkey][ddkey].dtype).value
                            )
                        continue

                    raise TypeError(
                        "Group data fields must be of type Dataset or dict of Dataset."
                    )

            return self

    def __getitem__(self, key):
        """Overloads indexing to retrieve elements in groups."""
        return self.groups.__getitem__(key)

    def __iter__(self):
        """Overloads iter to iterate over elements in groups."""
        return self.groups.items().__iter__()

    def update(self, **groups):
        """Updates groups in the datastore, overwriting past values."""
        for k, v in groups.items():
            self.groups[k] = v

    def add(self, **groups):
        """Adds a new groups to the datastore."""

        existing_keys = set(groups.keys()).intersection(set(self.groups.keys()))
        if existing_keys:
            raise ValueError(
                f"Keys {existing_keys} already exist in the datastore, use `update` instead if intending to overwrite past data."
            )

        self.update(**groups)


# %%
