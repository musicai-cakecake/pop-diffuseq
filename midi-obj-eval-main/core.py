import numpy as np
import mido
import pretty_midi
from norm_test import norm_l2_for_each_row, norm_l2_for_all_event
from miditoolkit.midi.parser import MidiFile


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def extract_pretty_midi_features(midi_filepath):
    return pretty_midi.PrettyMIDI(midi_filepath)


def extract_pretty_midi_features_multiple(midi_filepaths):
    return [extract_pretty_midi_features(midi_filepath) for midi_filepath in midi_filepaths]


def get_num_notes(pretty_midi_features):
    piano_roll = pretty_midi_features.instruments[0].get_piano_roll(fs=100)
    return piano_roll.sum()


def get_num_notes_one_instruments(pretty_midi_features):
    one_instruments = pretty_midi_features.instruments[0]
    piano_roll = one_instruments.get_piano_roll(fs=100)
    return piano_roll.sum()


def get_used_pitch(pretty_midi_features):
    """
    total_used_pitch (Pitch count): The number of different pitches within a sample.

    Returns:
    'used_pitch': pitch count, scalar for each sample.
    """
    piano_roll = pretty_midi_features.instruments[0].get_piano_roll(fs=100)
    sum_notes = np.sum(piano_roll, axis=1)
    used_pitch = np.sum(sum_notes > 0)
    return used_pitch


def get_used_pitch_one_instrument(pretty_midi_features):
    """
    total_used_pitch (Pitch count): The number of different pitches within a sample.

    Returns:
    'used_pitch': pitch count, scalar for each sample.
    """
    piano_roll = pretty_midi_features.instruments[0].get_piano_roll(fs=100)
    sum_notes = np.sum(piano_roll, axis=1)
    used_pitch = np.sum(sum_notes > 0)
    return used_pitch


def get_used_pitch_multiple(list_of_pretty_midi_features):
    sum_notes = 0
    for pretty_midi_features in list_of_pretty_midi_features:
        piano_roll = pretty_midi_features.instruments[0].get_piano_roll(fs=100)
        sum_notes += np.sum(piano_roll, axis=1)
    used_pitch = np.sum(sum_notes > 0)
    return used_pitch


def get_pitch_class_histogram(pretty_midi_features):
    """
    total_pitch_class_histogram (Pitch class histogram):
    The pitch class histogram is an octave-independent representation of the pitch content with a dimensionality of 12 for a chromatic scale.
    In our case, it represents to the octave-independent chromatic quantization of the frequency continuum.

    Returns:
    'histogram': histrogram of 12 pitch, with weighted duration shape 12
    """
    piano_roll = pretty_midi_features.instruments[0].get_piano_roll(fs=100)
    histogram = np.zeros(12)
    for i in range(0, 128):
        pitch_class = i % 12
        pitch_sum_i = np.sum(piano_roll, axis=1)[i]
        histogram[pitch_class] += pitch_sum_i
    histogram = histogram / sum(histogram)
    return histogram


def get_pitch_class_transition_matrix(pretty_midi_features, normalize=0):
    """
    从行到列的转移矩阵
    pitch_class_transition_matrix (Pitch class transition matrix):
    The transition of pitch classes contains useful information for tasks such as key detection, chord recognition, or genre pattern recognition.
    The two-dimensional pitch class transition matrix is a histogram-like representation computed by counting the pitch transitions for each (ordered) pair of notes.

    Args:
    'normalize' : If set to 0, return transition without normalization.
                  If set to 1, normalizae by row.
                  If set to 2, normalize by entire matrix sum.
    Returns:
    'transition_matrix': shape of [12, 12], transition_matrix of 12 x 12.
    """
    transition_matrix = pretty_midi_features.get_pitch_class_transition_matrix()

    if normalize == 0:
        return transition_matrix
    elif normalize == 1:
        sums = np.sum(transition_matrix, axis=1)
        sums[sums == 0] = 1
        return transition_matrix / sums.reshape(-1, 1)
    elif normalize == 2:
        # return transition_matrix / sum(sum(transition_matrix))
        return norm_l2_for_all_event(transition_matrix)
    elif normalize == 3:
        return norm_l2_for_each_row(transition_matrix)
    else:
        print("invalid normalization mode, return unnormalized matrix")
        return transition_matrix


def probability_of_pitch_transition_destination(pretty_midi_features):
    print('计算运动到每个音高的比例percentage')
    transition_matrix = pretty_midi_features.get_pitch_class_transition_matrix()
    transition_matrix_sums = np.sum(transition_matrix, axis=0)  # 按列加
    transition_matrix_probability = transition_matrix_sums/np.sum(transition_matrix_sums)
    return transition_matrix_probability
    # 计算比例


def get_avg_ioi(pretty_midi_features):
    """
    avg_IOI (Average inter-onset-interval):
    To calculate the inter-onset-interval in the symbolic music domain, we find the time between two consecutive notes.

    Returns:
    'avg_ioi': a scalar for each sample.
    """
    onset = pretty_midi_features.get_onsets()  # 获得音乐中所有音轨所有音符的onset times
    ioi = np.diff(onset)
    avg_ioi = np.mean(ioi)
    return avg_ioi


def get_avg_ioi_beats(midi_file_path, beat_interval_threshold=4, excluded_chord=True):
    print('计算midi音乐的inter-onset-interval 起奏间隔 以节拍为单位')
    midi_objects = MidiFile(midi_file_path)
    # ticks_16th = midi_objects.ticks_per_beat / 4.0  # ticks_16th

    onset_dict = dict()

    for ins in midi_objects.instruments:

        if ins.name is None or ins.name == "":
            ins_name = 'unknown'
        else:
            ins_name = ins.name

        onset_beats_list = list()
        for note in ins.notes:
            onset_beats_list.append(note.start / midi_objects.ticks_per_beat)

        # onset_np = np.array(onset_beats_list)
        # print('onset_np.shape:', onset_np.shape)
        ioi_beat = np.diff(np.array(onset_beats_list))  # a[n]-a[n-1]
        # print('ioi_beat.shape:', ioi_beat.shape)
        filtered_ioi = ioi_beat[ioi_beat <= beat_interval_threshold]
        if excluded_chord:
            filtered_ioi = filtered_ioi[filtered_ioi > 0]

        onset_dict[ins_name] = filtered_ioi.mean()

    return onset_dict


def get_avg_duration_second(pretty_midi_features):
    dur_np = np.array([])
    for note in pretty_midi_features.instruments[0].notes:
        dur_np = np.append(dur_np, note.get_duration())
    return dur_np.mean()


def get_avg_duration_beats(midi_file_path):
    """

    Args:
        midi_file_path: '.midi','.mid' files

    Returns: The average note duration in beats unit. One beat equal to the duration of quarter note.

    """
    midi_objects = MidiFile(midi_file_path)
    # ticks_16th = midi_objects.ticks_per_beat / 4.0
    ticks_32th = midi_objects.ticks_per_beat / 8.0

    avg_duration_beats = {}

    for ins in midi_objects.instruments:
        if ins.name is None or ins.name == "":
            ins_name = 'unknown'
        else:
            ins_name = ins.name

        dur_list = list()

        for note in ins.notes:
            dur = note.end - note.start
            if dur > ticks_32th:
                dur_list.append(dur)

        dur_np = np.array(dur_list)

        avg_duration_beats[ins_name] = dur_np.mean() / midi_objects.ticks_per_beat

    return avg_duration_beats

    # duration_ticks = {
    #     '16th': int(ticks_16th * 1),
    #     '8th': int(ticks_16th * 2),
    #     'three fourths': int(ticks_16th * 3),
    #     'one beat': int(ticks_16th * 4),
    #     'one beat half': int(ticks_16th * 6),
    #     'two beat': int(ticks_16th * 8),
    #     'three beat': int(ticks_16th * 12),
    #     'four beat': int(ticks_16th * 16),
    # }
