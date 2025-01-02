import os
from miditoolkit.midi.parser import MidiFile
from midi_quantize import duration_quantize_16th, check_track_pitch_between_c1_b6
from midi_quantize import onset_quantize_of_notes_controls_on_32th, check_note_number, check_instrument_number


if __name__ == '__main__':
    # folder_name = 'midi_data/lmd_aligned'
    folder_name = 'midi_data/lmd_matched_data/lmd_matched'

    # save_folder_path = 'midi_data/lmd_aligned_preprocessed_step1'
    save_folder_path = 'midi_data/lmd_matched_data/lmd_matched_preprocessed_step1'

    midi_paths = []
    dataset_midi_number = 0
    for root, dirs, files in os.walk(folder_name):
        if files:
            for file in files:
                if '.mid' in file:
                    dataset_midi_number += 1
                    midi_paths.append(os.path.join(root, file))

    processed_number = 0
    for midi_path in midi_paths:
        midi_file_name = midi_path.split('/')[-1]
        if '.mid' not in midi_file_name:
            print('not mid file')
            continue

        try:
            midi_raw = MidiFile(midi_path)
        except:
            print('cannot process midi file', midi_path)
            continue
        else:
            # check instrument number is existing
            # print('processing midi file', midi_file_name)

            if not check_instrument_number(midi_raw):
                print('wrong_program or lots of piano')
                continue
            else:

                # quantize
                # quantize onset
                onset_quantize_of_notes_controls_on_32th(midi_raw)

                # quantize duration
                quantized_midi = duration_quantize_16th(midi_raw)

                checked_midi = check_track_pitch_between_c1_b6(
                    quantized_midi, ['SOUND EFFECTS_ACCOMPANIMENT', 'PIANO_COMPOUND', 'REED_COMPOUND']
                )

                if len(checked_midi.instruments) < 3:
                    print('less than 3 tracks')
                    continue
                elif check_note_number(checked_midi) < 500:
                    print('less than 500 notes')
                    continue

                new_name = os.path.join(save_folder_path, midi_file_name)

                checked_midi.dump(new_name)
                processed_number += 1
                print(f'Processed {processed_number} midi')

                # try:
                #     checked_midi.dump(new_name)
                # except:
                #     print('cannot save midi file', new_name)
                # else:
                #     processed_number += 1
                #     print(f'Processed {processed_number} midi')
