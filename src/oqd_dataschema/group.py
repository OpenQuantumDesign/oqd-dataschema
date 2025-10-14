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

import typing
import warnings
from functools import reduce
from types import NoneType
from typing import Annotated, ClassVar, Literal, Union

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    TypeAdapter,
)

from oqd_dataschema.base import Attrs, GroupField
from oqd_dataschema.dataset import CastDataset

########################################################################################

__all__ = [
    "GroupBase",
    "GroupRegistry",
    "SinaraRawDataGroup",
    "MeasurementOutcomesDataGroup",
    "ExpectationValueDataGroup",
    "OQDTestbenchDataGroup",
]


########################################################################################


class GroupBase(BaseModel, extra="forbid"):
    """
    Schema representation for a group object within an HDF5 file.

    Each grouping of data should be defined as a subclass of `GroupBase`, and specify the datasets that it will contain.
    This base object only has attributes, `attrs`, which are associated to the HDF5 group.

    Attributes:
        attrs: A dictionary of attributes to append to the group.

    """

    attrs: Attrs = Field(default_factory=lambda: {})

    @staticmethod
    def _is_basic_groupfield_type(v):
        return reduce(
            lambda x, y: x or y,
            (gf._is_supported_type(v) for gf in GroupField.__subclasses__()),
        )

    @classmethod
    def _is_groupfield_type(cls, v):
        is_datafield = cls._is_basic_groupfield_type(v)

        is_annotated_datafield = typing.get_origin(
            v
        ) is Annotated and cls._is_basic_groupfield_type(v.__origin__)

        is_optional_datafield = typing.get_origin(v) is Union and (
            (v.__args__[0] == NoneType and cls._is_basic_groupfield_type(v.__args__[1]))
            or (
                v.__args__[1] == NoneType
                and cls._is_basic_groupfield_type(v.__args__[0])
            )
        )

        is_dict_datafield = (
            typing.get_origin(v) is dict
            and v.__args__[0] is str
            and cls._is_basic_groupfield_type(v.__args__[1])
        )

        return (
            is_datafield
            or is_annotated_datafield
            or is_optional_datafield
            or is_dict_datafield
        )

    @classmethod
    def _is_classvar(cls, v):
        return v is ClassVar or typing.get_origin(v) is ClassVar

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for k, v in cls.__annotations__.items():
            if k in ["class_", "attrs"]:
                raise AttributeError(
                    "`class_` and `attrs` attribute should not be set manually."
                )

            if cls._is_classvar(v):
                continue

            if not cls._is_groupfield_type(v):
                raise TypeError(
                    "All fields of `GroupBase` have to be of type `Dataset`, `Table` or `Folder`."
                )

        cls.__annotations__["class_"] = Literal[cls.__name__]
        setattr(cls, "class_", cls.__name__)

        # Auto-register new group types
        GroupRegistry.register(cls)


########################################################################################


class MetaGroupRegistry(type):
    """
    Metaclass for the GroupRegistry
    """

    def __new__(cls, clsname, superclasses, attributedict):
        attributedict["groups"] = dict()
        return super().__new__(cls, clsname, superclasses, attributedict)

    def register(cls, group):
        """Registers a group into the GroupRegistry."""
        if not issubclass(group, GroupBase):
            raise TypeError("You may only register subclasses of GroupBase.")

        if group.__name__ in cls.groups.keys():
            warnings.warn(
                f"Overwriting previously registered `{group.__name__}` group of the same name.",
                UserWarning,
                stacklevel=2,
            )

        cls.groups[group.__name__] = group

    def clear(cls):
        """Clear all registered types (useful for testing)"""
        cls.groups.clear()

    @property
    def union(cls):
        """Get the current Union of all registered types"""
        return Annotated[
            Union[tuple(cls.groups.values())], Discriminator(discriminator="class_")
        ]

    @property
    def adapter(cls):
        """Get TypeAdapter for current registered types"""
        return TypeAdapter(cls.union)


class GroupRegistry(metaclass=MetaGroupRegistry):
    """
    Represents the GroupRegistry
    """

    pass


########################################################################################


class SinaraRawDataGroup(GroupBase):
    """
    Example `Group` for raw data from the Sinara real-time control system.
    This is a placeholder for demonstration and development.
    """

    camera_images: CastDataset


class MeasurementOutcomesDataGroup(GroupBase):
    """
    Example `Group` for processed data classifying the readout of the state.
    This is a placeholder for demonstration and development.
    """

    outcomes: CastDataset


class ExpectationValueDataGroup(GroupBase):
    """
    Example `Group` for processed data calculating the expectation values.
    This is a placeholder for demonstration and development.
    """

    expectation_value: CastDataset


class OQDTestbenchDataGroup(GroupBase):
    """ """

    time: CastDataset
    voltages: CastDataset
