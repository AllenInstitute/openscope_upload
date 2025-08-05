import datetime
from aind_data_schema.core.instrument import Instrument
from aind_data_schema.components.devices import Objective, Detector, Laser, ImmersionMedium, DetectorType, DataInterface, Cooling
from aind_data_schema_models.organizations import Organization

instrument = Instrument(
    instrument_id="SLAP2_1",
    instrument_type="Two photon",
    manufacturer=Organization.AIND,
    modification_date=datetime.date(2025, 6, 16),
    objectives=[
        Objective(
            name="Leica Objective",
            numerical_aperture=1,
            magnification=20,
            manufacturer=Organization.LEICA,
            immersion=ImmersionMedium.WATER,
            serial_number="0119",
            model="507704"
        )
    ],
    detectors=[
        Detector(
            name="Hamamatsu Green Detector",
            detector_type=DetectorType.OTHER,
            manufacturer=Organization.HAMAMATSU,
            model="C13366-5286",
            serial_number="16D-001",
            data_interface=DataInterface.OTHER,
            cooling=Cooling.AIR,
            notes="Type and data interface are 'other' per rigDescription.json."
        ),
        Detector(
            name="Hamamatsu Red Detector",
            detector_type=DetectorType.OTHER,
            manufacturer=Organization.HAMAMATSU,
            model="C13366-1960",
            serial_number="22C-002",
            data_interface=DataInterface.OTHER,
            cooling=Cooling.AIR,
            notes="Type and data interface are 'other' per rigDescription.json."
        )
    ],
    light_sources=[
        Laser(
            name="Monaco150",
            manufacturer=Organization.COHERENT_SCIENTIFIC,
            serial_number="S0124263226",
            model="Monaco150",
            device_type="Laser",
            coupling="Free-space",
            wavelength=1035,
            maximum_power=150000
        )
    ],
    optical_tables=[],
    temperature_control=True,
    calibration_data="N/A",
    calibration_date=datetime.date(2023, 1, 4),
    notes="",
)

from aind_data_schema.base import AindCoreModel

with open("rig.json", "w") as f:
    f.write(AindCoreModel.model_dump_json(instrument, indent=4))
