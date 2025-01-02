import copy
import os
import json
import random

from more_itertools import split_before
from miditoolkit.midi.parser import MidiFile
from chords_detector.chords import ChordsDetector
from midi_quantize import pitch_time_duration_limit_detect

pit2alphabet = ['C', 'd', 'D', 'e', 'E', 'F', 'g', 'G', 'a', 'A', 'b', 'B']

all_track_name = []


def chord_progression_detect(midi_object):
    notes_whole = []
    ticks_32th = midi_object.ticks_per_beat / 8.0
    for instrument in midi_object.instruments:
        if not instrument.is_drum and 'DRUM' not in instrument.name and 'PERCUSSION' not in instrument.name:
            instrument.notes.sort(key=lambda x: (x.start, x.pitch))
            for note in instrument.notes:
                if round((note.end - note.start) / ticks_32th) >= 1:
                    notes_whole.append(note)

    notes_whole.sort(key=lambda x: (x.start, x.pitch))
    chords_event_per_beat = ChordsDetector(
        notes_whole, midi_object.ticks_per_beat
    ).detect_chords_by_chorder()
    # chords_event_per_beat
    chord_event_per_change = [chords_event_per_beat[0]]
    # chord_event_per_change
    for chord_event in chords_event_per_beat:
        if chord_event.value != chord_event_per_change[-1].value:
            chord_event_per_change.append(chord_event)

    max_chord_duration = int(ticks_32th * 64)

    chord_event_seq = [copy.copy(chord_event_per_change[0])]
    for i in range(1, len(chord_event_per_change)):
        if chord_event_per_change[i].time - chord_event_per_change[i - 1].time <= max_chord_duration:
            chord_event_seq.append(copy.copy(chord_event_per_change[i]))
        else:
            fixed_chord_event = copy.copy(chord_event_per_change[i])
            fixed_chord_event.time = chord_event_per_change[i - 1].time + max_chord_duration
            chord_event_seq.append(fixed_chord_event)
            chord_event_seq.append(copy.copy(chord_event_per_change[i]))
    notes_whole.clear()
    return chord_event_seq


def midi_to_event_sequence(p_midi):
    chord_seq = []
    note_seq = []
    instruments = []

    ticks_32th = p_midi.ticks_per_beat / 8.0  # ticks_32th
    ticks_16th = p_midi.ticks_per_beat / 4.0  # ticks_16th

    midi_chord = chord_progression_detect(p_midi)

    for chord_event in midi_chord:
        chord_seq.append([
            "0_CHORD",
            round(chord_event.time / ticks_32th),
            # round(chord_event.time/p_midi.ticks_per_beat),
            chord_event.value,
            'reserve'
        ])

    for track_idx, instrument in enumerate(p_midi.instruments):

        # process instrument name
        track_name = 'UNKNOWN'

        if 'melody' in instrument.name.lower():
            track_name = '0_MELODY'

        elif 'accompaniment' in instrument.name.lower():

            if instrument.name == 'STRING_ACCOMPANIMENT':
                track_name = 'STRINGS_ACCOMPANIMENT'

            elif instrument.name == 'SYNTH LEAD_ACCOMPANIMENT':
                track_name = 'SYNTH_LEAD_ACCOMPANIMENT'

            elif instrument.name == 'SYNTH PAD_ACCOMPANIMENT':
                track_name = 'SYNTH_PAD_ACCOMPANIMENT'

            elif instrument.name == 'CHROMATIC PERCUSSION_ACCOMPANIMENT':
                track_name = 'CHROMATIC_PERCUSSION_ACCOMPANIMENT'

            elif instrument.name == 'SYNTH EFFECTS_ACCOMPANIMENT':
                track_name = 'SYNTH_EFFECTS_ACCOMPANIMENT'

            elif instrument.name in ['SOUND EFFECTS_ACCOMPANIMENT', 'PIANO_COMPOUND', 'REED_COMPOUND']:
                continue
            else:
                track_name = instrument.name
        else:
            continue

        instruments.append(track_name)

        # process note information

        all_notes = instrument.notes
        all_notes.sort(key=lambda x: (x.start, x.pitch))

        for note in all_notes:

            # tick to beat
            note_dur = int(round((note.end - note.start) / ticks_16th))
            dur_fix = note_dur

            if note_dur == int(0):
                dur_fix = int(1)
            elif note_dur == int(5):
                dur_fix = int(4)
            elif note_dur == int(7):
                dur_fix = int(6)
            elif note_dur in [9, 10, 11, 12]:
                dur_fix = int(8)
            elif note_dur in [13, 14, 15] or note_dur > 16:
                dur_fix = int(16)

            if dur_fix not in [1, 2, 3, 4, 6, 8, 16]:
                dur_fix = int(1)

            note_seq.append([
                track_name,
                round(note.start / ticks_32th),
                note.pitch,
                dur_fix,
            ]
            )

    note_seq.sort(key=lambda x: (x[1], x[0]))

    return note_seq, instruments, chord_seq


def calculate_relative_time(seq: list, target_word: str):
    relative_time_seq = copy.deepcopy(seq)

    # Calculate the time of the note relative to the last chord
    last_chord_time = 0
    for eve in relative_time_seq:
        if eve[0] == target_word:
            last_chord_time = eve[1]
        else:
            eve[1] = eve[1] - last_chord_time

    # Calculate the time of the chord relative to the last chord
    last_chord_time_ids = 0
    for ids in range(len(relative_time_seq)):
        if relative_time_seq[ids][0] == target_word:
            relative_time_seq[ids][1] = seq[ids][1] - seq[last_chord_time_ids][1]
            last_chord_time_ids = ids
    return relative_time_seq


def get_tempo_token(midi_path):
    midi_object = MidiFile(midi_path)
    if midi_object.tempo_changes[0]:
        tempo = int(midi_object.tempo_changes[0].tempo)
    else:
        tempo = 120
    tempo_type = {
        'Tempo_Largo': [0, 65],
        'Tempo_Andante': [66, 108],
        'Tempo_Moderato': [109, 126],
        'Tempo_Allegro': [127, 168],
        'Tempo_Presto': [169, 999]
    }

    for tempo_name, tempo_bpm in tempo_type.items():
        if tempo_bpm[0] <= tempo <= tempo_bpm[1]:
            return tempo_name
    return 'Tempo_Moderato'


def midi_to_representation(midi_path):
    ori_midi = MidiFile(midi_path)

    note_sequence, instrument, chord_sequence = midi_to_event_sequence(ori_midi)

    note_sequence.extend(chord_sequence)
    note_sequence.sort(key=lambda x: (x[1], x[0]))

    # sort within chord interval
    split_by_chord = list(split_before(note_sequence, lambda x: x[0] == "0_CHORD"))

    representation_sort_within_chord = []

    for chord_split in split_by_chord:
        chord_split.sort(key=lambda x: (x[0], x[1]))
        representation_sort_within_chord.extend(chord_split)

    relative_time_representation = calculate_relative_time(representation_sort_within_chord, '0_CHORD')

    # trim to 5 instruments
    instrument = list(set(instrument))
    instrument.sort()
    # if 'A_MELODY' in instrument:
    #     instrument.remove('A_MELODY')
    while len(instrument) != 5:
        if len(instrument) > 5:
            instrument.pop()
        elif len(instrument) < 5:
            instrument.append('UNKNOWN')

    return relative_time_representation, instrument


def pit2str(x):
    octave = x // 12
    octave = octave - 1 if octave > 0 else 'L'
    rel_pit = x % 12
    return pit2alphabet[rel_pit] + str(octave)


def midi_tokenize(midi_attribution_seq: list):
    """
    input:

    """

    token_seq = []
    for event in midi_attribution_seq:

        if event[0] == '0_CHORD':
            token_seq.append(
                [
                    event[0],
                    'time' + str(event[1]),
                    'chord_' + event[2],
                    event[3]
                ]
            )
        else:
            token_seq.append(
                [
                    event[0],
                    'time' + str(event[1]),
                    pit2str(event[2]),
                    'dur' + str(event[3]),
                ]
            )

    return token_seq


def save_json_for_diffusion_model(midi_name: str, src_seq: list, trg_seq: list, json_name: str):
    # list to str
    if len(src_seq) == 7 and len(trg_seq) == 125:

        src_events = ""
        trg_events = ""

        for src_event in src_seq:
            src_events += (str(src_event) + " ")

        for t_event in trg_seq:
            for event in t_event:
                trg_events += (str(event) + " ")

        src_events.strip()
        trg_events.strip()

        one_midi_unit = {
            "name": midi_name,
            "src": src_events,
            "trg": trg_events
        }

        with open(json_name, 'a', encoding='utf-8') as f:
            json.dump(one_midi_unit, f)
            f.write('\n')

    else:
        print('the length of processed midi representation is fault! ')


def save_json_fragment_representation(
        midi_name: str, instrument_names: list, src_seq: list, trg_seq: list, json_name: str):
    # list to str

    # trg_seq_1D = [element for sublist in trg_seq for element in sublist]
    if len(trg_seq) > 500:
        frag_name_1 = midi_name+'_fragment_1'
        frag_name_2 = midi_name+'_fragment_2'
        frag_name_3 = midi_name+'_fragment_3'
        frag_name_4 = midi_name+'_fragment_4'

        src_seq_1 = copy.deepcopy(src_seq)
        src_seq_2 = copy.deepcopy(src_seq)
        src_seq_3 = copy.deepcopy(src_seq)
        src_seq_4 = copy.deepcopy(src_seq)

        src_seq_1.extend(instrument_names)
        src_seq_1.append('fragment_1')
        src_seq_2.extend(instrument_names)
        src_seq_2.append('fragment_2')
        src_seq_3.extend(instrument_names)
        src_seq_3.append('fragment_3')
        src_seq_4.extend(instrument_names)
        src_seq_4.append('fragment_4')

        trg_seq_1 = trg_seq[:125]
        trg_seq_2 = trg_seq[125:250]
        trg_seq_3 = trg_seq[250:375]
        trg_seq_4 = trg_seq[375:500]

        save_json_for_diffusion_model(frag_name_1, src_seq_1, trg_seq_1, json_name)
        save_json_for_diffusion_model(frag_name_2, src_seq_2, trg_seq_2, json_name)
        save_json_for_diffusion_model(frag_name_3, src_seq_3, trg_seq_3, json_name)
        save_json_for_diffusion_model(frag_name_4, src_seq_4, trg_seq_4, json_name)
    else:
        print('less than 500 notes')


def midi_to_jsonl(midi_file, jsonl_name):
    # print('process: ', file_path)
    # read music event from midi file

    midi_name = midi_file.split('/')[-1]

    midi_representation, ins = midi_to_representation(midi_file)

    had_long_rest, had_low_pitch, had_long_duration, had_short_len = pitch_time_duration_limit_detect(
        midi_representation, pitch_threshold=[24, 95], rest_threshold=96, duration_threshold=16
    )

    # had_long_rest =False
    # had_low_pitch = False
    # had_long_duration = False
    # had_short_len = False

    if had_long_rest:
        print('had long rest')
    elif had_low_pitch:
        print('had low pitch')
    elif had_long_duration:
        print('had long duration')
    elif had_short_len:
        print('had short len')
    else:
        # midi event to token
        midi_tokens = midi_tokenize(midi_representation)
        midi_tempo_token = [get_tempo_token(midi_file)]
        save_json_fragment_representation(midi_name, ins, midi_tempo_token, midi_tokens, jsonl_name)


if __name__ == '__main__':

    midi_files = []
    # pop909_files_path = 'midi_data/lmd_aligned_processed'
    pop909_files_path = 'midi_data/lmd_matched_data/lmd_matched_processed'

    for dirpath, dirnames, filenames in os.walk(pop909_files_path):
        for filename in filenames:
            if '.mid' in filename:
                midi_files.append(os.path.join(dirpath, filename))
    random.shuffle(midi_files)
    for file_path in midi_files[:14000]:
        midi_to_jsonl(file_path, 'tokenized_data/lmd_matched/train.jsonl')
    for file_path in midi_files[14000:]:
        midi_to_jsonl(file_path, 'tokenized_data/lmd_matched/valid.jsonl')
    for file_path in midi_files[14100:]:
        midi_to_jsonl(file_path, 'tokenized_data/lmd_matched/test.jsonl')

    # midi_to_jsonl('midi_data/test/score-demo.mid', 'tokenized_data/test/test.jsonl')

    print('finish!')
