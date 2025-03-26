from oqd_dataschema.base import Dataset, Group


class SinaraRawDataGroup(Group):
    camera_images: Dataset


class MeasurementOutcomesDataGroup(Group):
    outcomes: Dataset


class ExpectationValueDataGroup(Group):
    expectation_value: Dataset
