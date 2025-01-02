# -*- coding: utf-8 -*-
"""Helper functions
"""
import os
import copy
import numpy as np
import bisect
from miditoolkit import MidiFile, Instrument
from _collections import OrderedDict
from .constants import *
from collections import Counter


def dim(nested_list):
    """Get dimension for nested lists
    """
    if not type(nested_list) == list:
        return []
    return [len(nested_list)] + dim(nested_list[0])


# util functions for chord extraction using chorder
def get_pitch_map(midi_obj):
    """Get the distribution of pitch class, distribution is based on the frequency of pitch class
    :param midi_obj: MidiFile or Instrument or Instrument.notes
    :return distribution: pitch class distribution of midi_obj
    """
    distribution = [0] * 12
    if isinstance(midi_obj, MidiFile):
        for instrument in midi_obj.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                distribution[note.pitch % 12] += 1
    elif isinstance(midi_obj, Instrument):
        for note in midi_obj.notes:
            distribution[note.pitch % 12] += 1
    else:
        for note in midi_obj:
            distribution[note.pitch % 12] += 1
    return distribution


def get_key_from_pitch(midi_obj):
    """Get key from pitch by moving pitch class distribution and compare with the same diatonic pitches.
    """
    pitch_weights = [1, 1, 1, 1, 1, 1, 1]
    distribution = get_pitch_map(midi_obj)
    max_score = -1
    max_i = -1
    for i in range(12):
        temp_distribution = distribution[i:] + distribution[:i]
        score = sum([temp_distribution[pitch] * weight for pitch, weight in zip(DIATONIC_PITCHES, pitch_weights)])
        if score > max_score:
            max_score = score
            max_i = i
    return max_i


def get_key_from_pitch2(midi_obj):
    """The equivalent implementation as above, compare pitch class distribution with different diatonic pitches.
    """
    pitch_weights = [1, 1, 1, 1, 1, 1, 1]
    distribution = get_pitch_map(midi_obj)
    max_score = -1
    max_i = -1
    for i in range(12):
        temp_distribution = distribution
        pitches = [(p + i) % 12 for p in DIATONIC_PITCHES]
        score = sum([temp_distribution[pitch] * weight for pitch, weight in zip(pitches, pitch_weights)])
        if score > max_score:
            max_score = score
            max_i = i
    return max_i


def get_chord_map(notes, start=0, end=1e7):
    """Get the distribution of pitch class based on the accumulation of note duration
    """
    chord_map = [0] * 12
    for note in notes:
        chord_map[note.pitch % 12] += min(end, note.end) - max(start, note.start)
    return chord_map


# time points utils
def get_timepoints(midi_obj):
    """Get the notes starting at the same time
    If midi_obj is MidiFile, use the notes in all the tracks, otherwise, use only one track
    """
    timepoints = {}
    notes = []
    # Gather all the notes in the midi file
    if isinstance(midi_obj, MidiFile):
        for instrument in midi_obj.instruments:
            for note in instrument.notes:
                start = note.start
                end = note.end
                new_note = note
                new_note.dur = end - start

                if note.start in timepoints.keys():
                    timepoints[start].append(new_note)
                else:
                    timepoints[start] = [new_note]
    elif isinstance(midi_obj, Instrument):  # used for a single track
        notes = midi_obj.notes
    else:
        notes = midi_obj

    for note in notes:
        start = note.start
        end = note.end
        new_note = note
        new_note.dur = end - start

        if note.start in timepoints.keys():
            timepoints[start].append(new_note)
        else:
            timepoints[start] = [new_note]

    return OrderedDict(sorted(timepoints.items()))


def get_notes_at(timepoints, start=0, end=1e7):
    """Get the notes in timepoints using binary search
    """
    notes = []
    l_index = 0
    r_index = len(timepoints)

    while l_index < r_index:
        m = int((l_index + r_index) / 2)
        if list(timepoints.keys())[m] < start:
            l_index = m + 1
        else:
            r_index = m
    start_index = max(l_index - 10, 0)  # TODO: is this reasonable? start_index = max(l_index, 0)

    l_index = 0
    r_index = len(timepoints)
    while l_index < r_index:
        m = int((l_index + r_index) / 2)
        if list(timepoints.keys())[m] < end:
            l_index = m + 1
        else:
            r_index = m
    end_index = l_index

    partial_timepoints = list(timepoints.items())[start_index:end_index]
    for start_time, timepoint_notes in partial_timepoints:
        for note in timepoint_notes:
            end_time = start_time + note.dur
            if start_time < end and end_time >= start:
                notes.append(note)

    return notes


def get_notes_by_beats(midi_obj, beats=1, ticks_per_beat=480):
    """Get the notes within N beats
    :param midi_obj: MidiFile or Instrument or List[Note]
    :param beats: Number of beats to consider
    :param ticks_per_beat: beat resolutio in ticks
    :return: list of notes separated every N beats
    """
    tick_interval = ticks_per_beat * beats
    timepoints = get_timepoints(midi_obj)
    notes = []
    if isinstance(midi_obj, MidiFile):
        max_tick = midi_obj.max_tick
    elif isinstance(midi_obj, Instrument):
        tmp_notes = copy.deepcopy(midi_obj.notes)
        tmp_notes.sort(key=lambda x: x.end)
        max_tick = tmp_notes[-1].end
    else:
        tmp_notes = copy.deepcopy(midi_obj)
        tmp_notes.sort(key=lambda x: x.end)
        max_tick = tmp_notes[-1].end

    for tick_time in range(0, max_tick, tick_interval):
        notes.append(get_notes_at(timepoints, tick_time, tick_time + tick_interval))

    return notes


def get_chord_quality(notes, start=0, end=1e7):
    """Get chord quality with root note based on score.
    """
    max_score = 0
    max_root = -1
    max_quality = None
    chord_map = get_chord_map(notes, start, end)
    # A chord contains at least 3 different notes
    if Counter(chord_map)[0] >= 10:  # 10 = 12 pitch class minus 2
        return NO_CHORD, -1

    for i in range(12):
        temp_map = chord_map[i:] + chord_map[:i]
        if temp_map[0] == 0:
            continue
        for quality, weights in CHORD_WEIGHTS.items():
            score = sum([map_item * weight for map_item, weight in zip(temp_map, weights)])
            if score > max_score:
                max_score = score
                max_root = i
                max_quality = quality
    # If max_score is not evident enough
    if max_score < (end - start) * 10:
        return NO_CHORD, -1

    chord_name = str(max_root) + '_' + str(max_quality)

    return chord_name, max_score


def get_chords(midi_obj, beats=2, ticks_per_beat=480):
    """Get chords every N beats
    :param midi_obj: MidiFile or Instrument
    :param beats: N beats to consider at each step
    :param ticks_per_beat: Beat resolution in ticks
    :return: list of (chord_name, max_score) for each beats
    """
    interval = ticks_per_beat * beats
    res = [get_chord_quality(notes, i * interval, (i + 1) * interval) for i, notes in
           enumerate(get_notes_by_beats(midi_obj, beats, ticks_per_beat))]

    return [(chord[0], int(chord[1] / beats)) for chord in res]  # normalize score according to number of beats


# util functions for chord extraction using magenta chord extraction method
class ChordInferenceError(Exception):
    pass


class EmptySequenceError(ChordInferenceError):
    pass


class SequenceTooLongError(ChordInferenceError):
    pass


def get_frames_pc_vectors(notes, ticks_per_frame):
    """Compute pitch class vectors for temporal frames across a sequence.
    :param notes: List of notes, should be ordered
    :param ticks_per_frame: Number of ticks per frame(chord), ticks_per_frame = int(ticks_per_beat * 4 / chords_per_bar)
    """
    num_frames = int(notes[-1].end / ticks_per_frame)
    # Don't include initial start time and final end time
    frame_boundaries = ticks_per_frame * np.arange(1, num_frames)
    x = np.zeros([num_frames, 12])

    for note in notes:
        # bisect_right and bisect_left are not equivalent when the the element to be inserted is present in the list.
        # bisect.bisect_right returns the rightmost place in the sorted list to insert the given element.
        start_frame = bisect.bisect_right(frame_boundaries, note.start)
        # bisect.bisect_left returns the leftmost place in the sorted list to insert the given element.
        end_frame = bisect.bisect_left(frame_boundaries, note.end)
        pitch_class = note.pitch % 12

        if start_frame >= end_frame:
            x[start_frame, pitch_class] += note.end - note.start
        else:
            # Deal with start and end frame separately
            x[start_frame, pitch_class] += frame_boundaries[start_frame] - note.start
            for frame in range(start_frame + 1, end_frame):
                x[frame, pitch_class] += frame_boundaries[frame] - frame_boundaries[frame - 1]
            x[end_frame, pitch_class] += note.end - frame_boundaries[end_frame - 1]

    x_norm = np.linalg.norm(x, axis=1)  # reduce by column
    nonzero_frames = x_norm > 0
    x[nonzero_frames, :] /= x_norm[nonzero_frames, np.newaxis]  # broadcasting for division

    return x


def get_chord_frame_log_likelihood(frames_pc_vectors, chord_note_concentration):
    """Log-likelihood of observing each frame of note pitches under each chord.
    :param frames_pc_vectors: [num_frames, pitch_class]
    :param chord_note_concentration: parameter to control the influence of `chord_frame_log_likelihood`
    CHORD_PITCH_VECTORS: [num_chords, pitch_class]
    """
    # TODO: log isn't actually used here, this step should be called get_frame_chord_similarities
    return chord_note_concentration * np.dot(frames_pc_vectors, CHORD_PITCH_VECTORS.T)


def get_key_chord_distribution(chord_pitch_out_of_key_prob):
    """Probability distribution over chords for each key.
    TIP: the idea is that chords' probability of existence is different under different key. E.g, (0, 4, 7) is more
    likely to exist under Ionian mode of diatonic pitches (0, 2, 4, 5, 7, 9 , 11) than (0, 1, 3)
    :return: [pitch_class, num_of_chords]
    """
    num_pitches_in_key = np.zeros([12, len(ALL_CHORDS)], dtype=np.int32)
    num_pitches_out_of_key = np.zeros([12, len(ALL_CHORDS)], dtype=np.int32)

    # For each key and chord, compute the number of chord notes in the key and the
    # number of chord notes outside the key.
    for key in range(12):
        # Different `modes` of scale, e.g., (0, 2, 4, 5, 7, 9 , 11)
        key_pitches = set((key + offset) % 12 for offset in DIATONIC_PITCHES)
        for i, chord in enumerate(ALL_CHORDS[1:]):  # exclude NO_CHORD
            root, kind = chord
            # pitch class distribution of different chord qualities under different root pitch, e.g., [0, 4, 7]
            chord_pitches = set((root + offset) % 12 for offset in CHORD_MAPS[kind])
            num_pitches_in_key[key, i + 1] = len(chord_pitches & key_pitches)
            num_pitches_out_of_key[key, i + 1] = len(chord_pitches - key_pitches)

    # Compute the probability of each chord under each key, normalizing to sum to
    # one for each key.
    mat = ((1 - chord_pitch_out_of_key_prob) ** num_pitches_in_key *
           chord_pitch_out_of_key_prob ** num_pitches_out_of_key)
    mat /= mat.sum(axis=1)[:, np.newaxis]
    return mat


def get_key_chord_transition_distribution(key_chord_distribution, key_change_prob, chord_change_prob):
    """Transition distribution between key-chord pairs."""
    mat = np.zeros([len(ALL_KEY_CHORDS), len(ALL_KEY_CHORDS)])

    for i, key_chord_1 in enumerate(ALL_KEY_CHORDS):
        key_1, chord_1 = key_chord_1
        chord_index_1 = i % len(ALL_CHORDS)

        for j, key_chord_2 in enumerate(ALL_KEY_CHORDS):
            key_2, chord_2 = key_chord_2
            chord_index_2 = j % len(ALL_CHORDS)

            if key_1 != key_2:
                # Key change. Chord probability depends only on key and not previous chord.
                mat[i, j] = (key_change_prob / 11)
                mat[i, j] *= key_chord_distribution[key_2, chord_index_2]

            else:
                # No key change.
                mat[i, j] = 1 - key_change_prob
                if chord_1 != chord_2:
                    # Chord probability depends on key, but we have to redistribute the
                    # probability mass on the previous chord since we know the chord changed.
                    mat[i, j] *= (chord_change_prob * (key_chord_distribution[key_2, chord_index_2] +
                                                       key_chord_distribution[key_2, chord_index_1] /
                                                       (len(ALL_CHORDS) - 1)))
                else:
                    # No chord change.
                    mat[i, j] *= 1 - chord_change_prob

    return mat


def perform_key_chord_viterbi(frame_chord_loglik, key_chord_loglik, key_chord_transition_loglik):
    """Use the Viterbi algorithm to infer a sequence of key-chord pairs.
    :return: [(0, (9, 'min')), (0, (9, 'min')), (0, (9, 'min')), (0, (9, 'min'))]
    """
    num_frames, num_chords = frame_chord_loglik.shape
    num_key_chords = len(key_chord_transition_loglik)

    loglik_matrix = np.zeros([num_frames, num_key_chords])
    path_matrix = np.zeros([num_frames, num_key_chords], dtype=np.int32)

    # Initialize with a uniform distribution over keys.
    for i, key_chord in enumerate(ALL_KEY_CHORDS):
        key, unused_chord = key_chord
        chord_index = i % len(ALL_CHORDS)
        loglik_matrix[0, i] = (-np.log(12) + key_chord_loglik[key, chord_index] + frame_chord_loglik[0, chord_index])

    for frame in range(1, num_frames):
        # At each frame, store the log-likelihood of the best sequence ending in
        # each key-chord pair, along with the index of the parent key-chord pair
        # from the previous frame.
        mat = (np.tile(loglik_matrix[frame - 1][:, np.newaxis], [1, num_key_chords]) +
               key_chord_transition_loglik)
        path_matrix[frame, :] = mat.argmax(axis=0)
        loglik_matrix[frame, :] = (mat[path_matrix[frame, :], range(num_key_chords)] +
                                   np.tile(frame_chord_loglik[frame], 12))

    # Reconstruct the most likely sequence of key-chord pairs.
    path = [np.argmax(loglik_matrix[-1])]
    for frame in range(num_frames, 1, -1):
        path.append(path_matrix[frame - 1, path[-1]])

    return [(index // num_chords, ALL_CHORDS[index % num_chords]) for index in path[::-1]]


def findall_endswith(postfix, root):
    """Traverse `root` recursively and yield all files ending with `postfix`."""
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(postfix):
                yield os.path.join(dirpath, filename)


def get_tempo_class(tempos_mean):
    """Classify tempo into different bins.
    """
    if tempos_mean in DEFAULT_TEMPO_INTERVALS[0] or tempos_mean <= DEFAULT_TEMPO_INTERVALS[0].start:
        tempo = 0
    elif tempos_mean in DEFAULT_TEMPO_INTERVALS[1]:
        tempo = 1
    elif tempos_mean in DEFAULT_TEMPO_INTERVALS[2] or tempos_mean >= DEFAULT_TEMPO_INTERVALS[2].stop:
        tempo = 2
    return tempo
