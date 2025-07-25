from aind_metadata_mapper.stimulus.camstim import Camstim, CamstimSettings
import json
import pathlib as pl
import pandas as pd

experiment_info = json.loads(pl.Path(__file__).with_name('experiment_info.json').read_text())

session_settings = CamstimSettings(sessions_root=None,
                                    opto_conditions_map=None,
                                    overwrite_tables=False,
                                    mtrain_server='http://mtrain:5000',
                                    input_source=pl.WindowsPath('//allen/programs/mindscope/production/openscope/prod0/specimen_1365565234/ophys_session_1386983913'),
                                    output_directory=pl.WindowsPath('//allen/aind/scratch/2p-working-group/data-uploads/731327_2024-10-30_14-03-07'),
                                    session_id='1386983913',
                                    subject_id='731327')

print(session_settings)
session_mapper = Camstim(session_settings)
# session_mapper.build_stimulus_table

stim_table = pd.read_csv(session_mapper.stim_table_path)
print(len(stim_table))
epochs = session_mapper.extract_stim_epochs(stim_table)
with open('epochs_file.txt', 'a') as epochs_file:
    for epoch in epochs:
        epochs_file.write(str(epoch)+'\n')
print(epochs[:100])

# session_mapper.run_job()
