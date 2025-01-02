import copy
import os
from miditoolkit.midi.parser import MidiFile


def split_midi_by_markers(midi_object, target_markers: list, reserve_beat=int(2), fix_bass=True, split_one=True):

    reserve_beat_ticks = midi_object.ticks_per_beat * reserve_beat

    cut_time_list = []

    for target_marker in target_markers:
        for mark in midi_object.markers:
            if target_marker in mark.text:
                cut_time_list.append(mark.time)
                if split_one:
                    break

    split_midi_list = []
    if cut_time_list:
        for cut_time in cut_time_list:

            midi_copy = copy.deepcopy(midi_object)
            midi_copy.markers.clear()
            cut_time -= reserve_beat_ticks
            cut_time_tempo = get_tempo_by_input_tick(midi_copy, cut_time)
            set_tempo_by_input_tempo(midi_copy, cut_time_tempo)

            del_note_idx = []
            if cut_time > 0:
                for track_idx, instrument in enumerate(midi_copy.instruments):
                    instrument.control_changes.clear()
                    instrument.pedals.clear()
                    instrument.pitch_bends.clear()
                    del_note_idx.clear()
                    for note_idx, note in enumerate(instrument.notes):

                        if fix_bass and instrument.name == 'BASS_ACCOMPANIMENT':
                            note.pitch -= 24
                        if note.start < cut_time:
                            del_note_idx.append(note_idx)
                        else:
                            note.start -= cut_time
                            note.end -= cut_time
                    if del_note_idx:
                        for wrong_idx in reversed(del_note_idx):
                            instrument.notes.pop(wrong_idx)

                split_midi_list.append(midi_copy)

        if split_midi_list:
            return split_midi_list
        else:
            return [midi_object]

    else:
        return [midi_object]


def cut_intra(midi_object, target_track='melody', reserve_beat=int(2)):

    p_midi = copy.deepcopy(midi_object)

    # ticks_32th = p_midi.ticks_per_beat / 8.0  # ticks_32th
    reserve_section_ticks = reserve_beat * p_midi.ticks_per_beat

    fist_tick = int(0)
    first_note = []
    # find first note
    for ins in p_midi.instruments:
        if ins.name.lower() == target_track:
            ins.remove_invalid_notes(verbose=False)
            all_notes = ins.notes
            all_notes.sort(key=lambda x: (x.start, x.pitch))
            first_note.append(all_notes[0].start)
    if first_note:
        fist_tick = min(first_note)

    cut_ticks = fist_tick - reserve_section_ticks
    if cut_ticks > 0:
        for track_idx, instrument in enumerate(p_midi.instruments):
            del_note_idx = []
            instrument.notes.sort(key=lambda x: (x.start, x.pitch))
            for note_idx, note in enumerate(instrument.notes):
                if note.start < cut_ticks:
                    del_note_idx.append(note_idx)
                else:
                    note.start -= cut_ticks
                    note.end -= cut_ticks
            if del_note_idx:
                for wrong_idx in reversed(del_note_idx):
                    instrument.notes.pop(wrong_idx)
        return p_midi
    else:
        return midi_object


def get_tempo_by_input_tick(midi_object, in_tick):
    """
    return tempo by input tick time
    :param midi_object:
    :param in_tick: the time in tick you want to get the tempo
    :return: tempo
    """
    tempo_list = []
    tempo = 120
    for tempo_event in midi_object.tempo_changes:
        if in_tick > tempo_event.time:
            tempo = tempo_event.tempo
        else:
            break
    return tempo


def set_tempo_by_input_tempo(midi_object, in_tempo):
    """
    set music tempo by input tempo, one music one tempo
    :param midi_object:
    :param in_tick:
    :return:
    """
    midi_object.tempo_changes[0].tempo = in_tempo
    midi_object.tempo_changes[0].time = 0
    while len(midi_object.tempo_changes) != 1:
        midi_object.tempo_changes.pop()


def set_average_tempo(midi_object):
    p_midi = copy.deepcopy(midi_object)
    tempo_list = []

    for tempo in midi_object.tempo_changes:
        tempo_list.append(tempo.tempo)
    if tempo_list:
        average_tempo = float(round(sum(tempo_list)/len(tempo_list)))
    else:
        average_tempo = float(120)

    first_tempo = midi_object.tempo_changes[0]
    first_tempo.tempo = average_tempo
    first_tempo.time = int(0)
    p_midi.tempo_changes.clear()
    p_midi.tempo_changes = [first_tempo]

    for ins in p_midi.instruments:
        ins.control_changes.clear()
        ins.pedals.clear()

    return p_midi


def find_first_note_tick(midi_object):
    first_note = []
    for ins in midi_object.instruments:
        if ins.notes:
            ins.remove_invalid_notes(verbose=False)
            all_notes = ins.notes
            all_notes.sort(key=lambda x: x.start)
            first_note.append(all_notes[0].start)
    if first_note:
        fist_note_tick = min(first_note)
        return fist_note_tick
    else:
        return 0


def onset_quantize_of_notes_controls_on_32th(midi_object: MidiFile):
    nth_tick = midi_object.ticks_per_beat / 8.0  # ticks_32th
    # fist_tick = find_first_note_tick(midi_object)

    for instrument in midi_object.instruments:

        # quantize onset of note
        if instrument.notes:
            instrument.remove_invalid_notes(verbose=False)
            instrument.notes.sort(key=lambda x: (x.start, x.pitch))
            for note in instrument.notes:
                # note.start -= fist_tick
                # note.end -= fist_tick
                note.start = int(round(note.start/nth_tick) * nth_tick)
            instrument.remove_invalid_notes(verbose=False)

        # quantize control_changes
        if instrument.control_changes:
            for controls in instrument.control_changes:
                # controls.time -= fist_tick
                controls.time = int(round(controls.time/nth_tick) * nth_tick)


def midi_quantize(midi_object, qth_32=True):

    p_midi = copy.deepcopy(midi_object)
    if qth_32:
        nth_tick = p_midi.ticks_per_beat / 8.0  # ticks_32th
    else:
        nth_tick = p_midi.ticks_per_beat / 4.0  # ticks_16th

    for ins in p_midi.instruments:
        ins.remove_invalid_notes(verbose=False)

    fist_tick = find_first_note_tick(p_midi)

    for track_idx, instrument in enumerate(p_midi.instruments):
        wrong_note_idx = []
        instrument.notes.sort(key=lambda x: (x.start, x.pitch))
        for note_idx, note in enumerate(instrument.notes):
            note.start -= fist_tick
            note.end -= fist_tick
            if note.end - note.start < (nth_tick/2):
                wrong_note_idx.append(note_idx)
            else:
                # quantize
                note.start = int(round(note.start/nth_tick) * nth_tick)
                note.end = int(round(note.end/nth_tick) * nth_tick)

        if wrong_note_idx:
            for wrong_idx in reversed(wrong_note_idx):
                instrument.notes.pop(wrong_idx)
    for ins in p_midi.instruments:
        ins.remove_invalid_notes(verbose=False)

    return p_midi


def duration_quantize_16th(midi_object):
    p_midi = copy.deepcopy(midi_object)

    ticks_16th = p_midi.ticks_per_beat / 4.0  # ticks_16th

    # duration_ticks = {
    #     '16th': int(ticks_16th * 1),
    #     '8th': int(ticks_16th * 2),
    #     'three fourths': int(ticks_16th * 3),
    #     'one beat': int(ticks_16th * 4),
    #     'one beat half': int(ticks_16th * 6),
    #     'two beat': int(ticks_16th * 8),
    #     'four beat': int(ticks_16th * 16),
    # }

    for track_idx, ins in enumerate(p_midi.instruments):

        for note_idx, note in enumerate(ins.notes):
            dur = note.end - note.start
            number_16th = int(round(dur/ticks_16th))
            # dur_fix = number_16th
            if number_16th == int(0):
                number_16th = int(1)
            elif number_16th == int(5):
                number_16th = int(4)
            elif number_16th == int(7):
                number_16th = int(6)
            elif number_16th in [9, 10, 11, 12]:
                number_16th = int(8)
            elif number_16th in [13, 14, 15] or number_16th > 16:
                number_16th = int(16)

            if number_16th in [1, 2, 3, 4, 6, 8, 16]:
                note.end = int(note.start + number_16th * ticks_16th)
            else:
                note.end = int(note.start + ticks_16th)

    return p_midi


def duration_ticks_to_16th(dur_in_ticks: int, ticks_per_beat: int):
    ticks_16th = ticks_per_beat / 4.0  # ticks_16th
    note_dur = int(round(
        dur_in_ticks / (ticks_per_beat / 4.0)
    ))
    if note_dur == int(0):
        note_dur = int(1)
    elif note_dur == int(5):
        note_dur = int(4)
    elif note_dur == int(7):
        note_dur = int(6)
    elif note_dur in [9, 10, 11, 12]:
        note_dur = int(8)
    elif note_dur in [13, 14, 15] or note_dur > 16:
        note_dur = int(16)
    if note_dur not in [1, 2, 3, 4, 6, 8, 16]:
        note_dur = int(1)
    return note_dur


def pitch_time_duration_limit_detect(music_seq, pitch_threshold, rest_threshold, duration_threshold, len_threshold=61):
    had_rest = False
    had_low_pitch = False
    had_long_duration = False
    had_short_len = False
    for event in music_seq:
        if event[1] > rest_threshold:
            had_rest = True
        if event[0] != '0_CHORD' and pitch_threshold[0] > event[2] > pitch_threshold[1]:
            had_low_pitch = True
        if event[0] != '0_CHORD' and event[3] > duration_threshold:
            had_long_duration = True
    if len(music_seq) < len_threshold:
        had_short_len = True
    return had_rest, had_low_pitch, had_long_duration, had_short_len


def check_note_number(midi_file: MidiFile):
    note_number = 0
    for track in midi_file.instruments:
        note_number += len(track.notes)
    return note_number


def check_instrument_number(midi_file: MidiFile):
    ins_program = []
    for instrument in midi_file.instruments:
        if instrument.program != 0 and not instrument.is_drum:
            ins_program.append(instrument.program)

    if len(ins_program) > 2:
        return True
    else:
        return False


def check_track_pitch_between_c1_b6(object_midi, del_track_name: list):


    false_track_id = []

    for idx, ins in enumerate(object_midi.instruments):

        if ins.name in del_track_name:
            false_track_id.append([idx, ins.name])

        elif ins.notes:
            for note in ins.notes:
                if note.pitch < 24 or note.pitch > 95:
                    false_track_id.append([idx, ins.name])
                    break
        else:
            false_track_id.append([idx, ins.name])

    if false_track_id:
        for wrong_ins in reversed(false_track_id):
            if object_midi.instruments[int(wrong_ins[0])].name == wrong_ins[1]:
                object_midi.instruments.pop(wrong_ins[0])
    return object_midi


if __name__ == '__main__':
    midi_files = []
    pop909_files_path = 'midi_data/POP909'
    for dirpath, dirnames, filenames in os.walk(pop909_files_path):
        for filename in filenames:
            if '.mid' in filename and 'v' not in filename:
                midi_files.append(os.path.join(dirpath, filename))

    for file_path in midi_files[66:67]:

        print('process: ', file_path)

        ori_midi = MidiFile(file_path)

        quantized_midi = midi_quantize(ori_midi)

        quantized_midi.dump('quantized_midi.mid')

    print('done')
