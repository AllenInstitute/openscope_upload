from aind_metadata_mapper.mesoscope.session import MesoscopeEtl

job_settings = r"""
    {"experimenter_full_name":["John Smith","Jane Smith"],
    "subject_id":"12345",
    "session_start_time":"2023-10-10T10:10:10",
    "session_end_time":"2023-10-10T18:10:10",
    "project":"my_project",
    "input_source":"\\\\allen\\aind\\scratch\\2p-working-group\\data-uploads\\multiplane-ophys_736963_2024-07-24_10-06-58\\ophys",
    "behavior_source":"\\\\allen\\aind\\scratch\\2p-working-group\\data-uploads\\multiplane-ophys_736963_2024-07-24_10-06-58\\ophys\\behavior",
    "output_directory":"."}
"""

from aind_metadata_mapper.mesoscope.models import JobSettings
from aind_metadata_mapper.mesoscope.session import MesoscopeEtl
from pathlib import Path
from datetime import datetime as dt

user_input = JobSettings(
    input_source=Path(r'\\allen\aind\scratch\2p-working-group\data-uploads\multiplane-ophys_736963_2024-07-24_10-06-58\ophys'),
    behavior_source=Path(r'\\allen\aind\scratch\2p-working-group\data-uploads\multiplane-ophys_736963_2024-07-24_10-06-58\behavior'),
    session_id="1382209121",
    output_directory=Path(r"."),
    session_start_time=dt.now(),
    session_end_time=dt.now(),
    subject_id="123456",
    project="Some Project",
    experimenter_full_name=["Bilbo Baggins"]
)

mesoscope = MesoscopeEtl(job_settings=user_input)
print(mesoscope)

mesoscope.run_job()