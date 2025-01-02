import os
import json

import copy
from miditoolkit.midi.parser import MidiFile
from midi_quantize import check_note_number, check_instrument_number


def read_track_analyse_result(program_result_json_path: str) -> dict:
    data_program_result = []

    with open(
            program_result_json_path,
            'r',
            encoding='utf-8'
    ) as fw:
        # injson = json.load(fw)
        for line in fw.readlines():
            dic = json.loads(line)
            data_program_result.append(dic)

    return data_program_result[0]


def assign_instrument_name(midi_file: MidiFile, melody_ins_program: int):

    instrument_category_and_program_dict = {
        'PIANO_ACCOMPANIMENT': [0, 7],
        'CHROMATIC_PERCUSSION_ACCOMPANIMENT': [8, 15],  # is_drum = False
        'ORGAN_ACCOMPANIMENT': [16, 23],
        'ACOUSTIC_GUITAR_ACCOMPANIMENT': [24, 25],
        'ELECTRIC_GUITAR_ACCOMPANIMENT': [26, 32],
        'BASS_ACCOMPANIMENT': [32, 39],
        'STRINGS_ACCOMPANIMENT': [40, 47],
        'ENSEMBLE_ACCOMPANIMENT': [48, 55],
        'BRASS_ACCOMPANIMENT': [56, 63],
        'REED_ACCOMPANIMENT': [64, 71],
        'PIPE_ACCOMPANIMENT': [72, 79],
        'SYNTH_LEAD_ACCOMPANIMENT': [80, 87],
        'SYNTH_PAD_ACCOMPANIMENT': [88, 95],
        'SYNTH_EFFECTS_ACCOMPANIMENT': [96, 103],
        'ETHNIC_ACCOMPANIMENT': [104, 111],
        'PERCUSSIVE_ACCOMPANIMENT': [112, 119],
        'SOUND_EFFECTS_ACCOMPANIMENT': [120, 127]
    }

    for instrument in midi_file.instruments:
        if 'melody' in instrument.name.lower():
            continue
        elif instrument.program == melody_ins_program:
            instrument.name = 'melody'
            continue
        elif instrument.is_drum:
            instrument.name = 'PERCUSSIVE_ACCOMPANIMENT'
        else:
            for ins_category, program_interval in instrument_category_and_program_dict.items():
                if program_interval[1] >= instrument.program >= program_interval[0]:
                    instrument.name = ins_category


def fix_tempo(midi_file: MidiFile):

    first_tempo = copy.deepcopy(midi_file.tempo_changes[0])

    if len(midi_file.tempo_changes) > 5:

        tempo_mean = 0
        for tempo in midi_file.tempo_changes[:5]:
            tempo_mean += tempo.tempo
        tempo_mean = tempo_mean / 5

        first_tempo.tempo = tempo_mean
        first_tempo.time = int(0)

        midi_file.tempo_changes = [first_tempo]

    else:
        midi_file.tempo_changes = [first_tempo]


if __name__ == '__main__':

    folder_name = 'midi_data/lmd_matched_data/lmd_matched_extract'
    save_folder_path = 'midi_data/lmd_matched_data/lmd_matched_processed'

    track_analyse_result = read_track_analyse_result('midi_data/lmd_matched_data/lmd_matched_extract/program_result.json')

    melody_track_label_dict = dict()
    for old_midi_path, labels_dict in track_analyse_result.items():
        midi_name = old_midi_path.split('/')[-1]
        if '.mid' in midi_name and midi_name not in melody_track_label_dict and labels_dict['melody']:
            melody_track_label_dict[midi_name] = labels_dict['melody']

    midi_paths = []
    for root, dirs, files in os.walk(folder_name):
        if files:
            for file in files:
                if '.mid' in file:
                    midi_paths.append(os.path.join(root, file))

    # assign track name
    for midi_path in midi_paths:
        midi_file_name = midi_path.split('/')[-1]
        if midi_file_name in melody_track_label_dict:
            melody_track_program = melody_track_label_dict[midi_file_name]
            try:
                midi_object = MidiFile(midi_path)
            except:
                print('cannot process midi file', midi_path)
                continue
            else:

                if check_note_number(midi_object) >= 500 and check_instrument_number(midi_object):

                    assign_instrument_name(midi_object, melody_track_program)
                    fix_tempo(midi_object)

                    midi_object.dump(
                        os.path.join(save_folder_path, midi_file_name)
                    )
                    print('write midi done, filename: ', midi_file_name)
                else:
                    print('')
                    continue

    print('done!')

