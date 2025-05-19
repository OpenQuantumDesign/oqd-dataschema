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

{ lib
, buildPythonPackage
, pythonOlder
, pydantic
, h5py
, bidict
, pytest
, pymdown-extensions
, mkdocstrings
, mkdocs-material
, mkdocstrings-python
, mdx-truly-sane-lists
}:

buildPythonPackage rec {
  pname = "oqd-dataschema";
  version = "0.1.0";
  format = "pyproject";

  disabled = pythonOlder "3.10";

  src = ./..;

  propagatedBuildInputs = [
    pydantic
    h5py
    bidict
  ];

  checkInputs = [
    pytest
  ];

  nativeBuildInputs = [
    pymdown-extensions
    mkdocstrings
    mkdocs-material
    mkdocstrings-python
    mdx-truly-sane-lists
  ];

  pythonImportsCheck = [ "oqd_dataschema" ];

  meta = with lib; {
    description = "OpenQuantum Design Data Schema";
    homepage = "https://github.com/OpenQuantumDesign/oqd-dataschema";
    license = licenses.asl20;
    maintainers = with maintainers; [ ];
  };
}
