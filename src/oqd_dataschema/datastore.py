import pathlib
from typing import Union

import h5py
import numpy as np
from pydantic import BaseModel

from oqd_dataschema.base import Dataset
from oqd_dataschema.groups import (
    ExpectationValueDataGroup,
    MeasurementOutcomesDataGroup,
    SinaraRawDataGroup,
)

GroupSubtypes = Union[
    SinaraRawDataGroup, MeasurementOutcomesDataGroup, ExpectationValueDataGroup
]


class Datastore(BaseModel):
    groups: dict[str, GroupSubtypes]

    def model_dump_hdf5(self, filepath: pathlib.Path):
        filepath.parent.mkdir(exist_ok=True, parents=True)

        with h5py.File(filepath, "a") as f:
            f.attrs["model"] = self.model_dump_json()
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
    def model_validate_hdf5(cls, filepath: pathlib.Path):
        with h5py.File(filepath, "r") as f:
            self = cls.model_validate_json(f.attrs["model"])
            for gkey, group in self.groups.items():
                for dkey, val in group.__dict__.items():
                    if dkey == "attrs":
                        continue
                    group.__dict__[dkey] = np.array(f[gkey][dkey][()])
            return self
