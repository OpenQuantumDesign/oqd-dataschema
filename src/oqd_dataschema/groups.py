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

from oqd_dataschema.base import Dataset, Group


class SinaraRawDataGroup(Group):
    """
    Example `Group` for raw data from the Sinara real-time control system. 
    This is a placeholder for demonstration and development.
    """
    camera_images: Dataset


class MeasurementOutcomesDataGroup(Group):
    """
    Example `Group` for processed data classifying the readout of the state. 
    This is a placeholder for demonstration and development.
    """
    outcomes: Dataset


class ExpectationValueDataGroup(Group):
    """
    Example `Group` for processed data calculating the expectation values.
    This is a placeholder for demonstration and development.
    """
    expectation_value: Dataset
