# -*- coding: utf-8 -*-
"""Chord detector class
"""
from typing import List
from collections import Counter
import numpy as np
from miditoolkit import Note
from .vocabulary import Event
from .constants import CHORD_MAPS, MAX_NUM_CHORDS, NO_CHORD
from .utils import get_chords, EmptySequenceError, SequenceTooLongError, get_frames_pc_vectors, \
    get_chord_frame_log_likelihood, get_key_chord_distribution, get_key_chord_transition_distribution, \
    perform_key_chord_viterbi


class ChordsDetector:
    def __init__(self, notes: List[Note], time_division: int):
        """ class constructor
        :param notes: notes to analyse (sorted by starting time, them pitch)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
        """
        self.notes = notes
        self.time_division = time_division

    def detect_chords_by_intervals(self, beat_res: int = 4, onset_offset: int = 1,
                                   only_known_chord: bool = False, simul_notes_limit: int = 20) -> List[Event]:
        """ Chord detection method based on pitch intervals
        NOTE: make sure to sort notes by start time then pitch before: notes.sort(key=lambda x: (x.start, x.pitch))
        NOTE2: on very large tracks with high note density this method can be very slow !
        If you plan to use it with the Maestro or GiantMIDI datasets, it can take up to
        hundreds of seconds per MIDI depending on your cpu.
        One time step at a time, it will analyse the notes played together
        and detect possible chords.

        :param beat_res: beat resolution, i.e. nb of samples per beat (default 4)
        :param onset_offset: maximum offset (in samples) ∈ N separating notes starts to consider them
                                starting at the same time / onset (default is 1)
        :param only_known_chord: will select only known chords. If set to False, non recognized chords of
                                n notes will give a chord_n event (default False)
        :param simul_notes_limit: nb of simultaneous notes being processed when looking for a chord
                this parameter allows to speed up the chord detection (default 20)
        :return: the detected chords as Event objects
        """
        assert simul_notes_limit >= 5, 'simul_notes_limit must be higher than 5, chords can be made up to 5 notes'
        tuples = []
        for note in self.notes:
            tuples.append((note.pitch, int(note.start), int(note.end)))
        notes = np.asarray(tuples)
        time_div_half = self.time_division // 2
        # samples to look at, if onset_offset = 1, this equals to ticks_per_sample
        onset_offset = self.time_division * onset_offset / beat_res
        count = 0
        previous_tick = -1
        chords = []
        while count < len(notes):
            # Checks we moved in time after last step, otherwise discard this tick
            if notes[count, 1] == previous_tick:
                count += 1
                continue

            # Gathers the notes around the same time step
            onset_notes = notes[count:count + simul_notes_limit]  # reduces the scope
            # onsets must be close enough, e.g., within several samples
            onset_notes = onset_notes[np.where(onset_notes[:, 1] <= onset_notes[0, 1] + onset_offset)]
            # If it is ambiguous, e.g. the notes lengths are too different
            if np.any(np.abs(onset_notes[:, 2] - onset_notes[0, 2]) > time_div_half):
                count += len(onset_notes)
                continue

            # Selects the possible chords notes
            if notes[count, 2] - notes[count, 1] <= time_div_half:
                onset_notes = onset_notes[np.where(onset_notes[:, 1] == onset_notes[0, 1])]
            chord = onset_notes[np.where(onset_notes[:, 2] - onset_notes[0, 2] <= time_div_half)]
            # Creates the "chord map" and see if it has a "known" quality, append a chord event if it is valid
            chord_map = tuple(chord[:, 0] - chord[0, 0])
            if 3 <= len(chord_map) <= 5 and chord_map[-1] <= 24:  # max interval between the root and highest degree
                chord_quality = len(chord)
                for quality, known_chord in CHORD_MAPS.items():
                    if known_chord == chord_map:
                        chord_quality = quality
                        break
                if only_known_chord and isinstance(chord_quality, int):
                    count += len(onset_notes)  # Move to the next notes
                    continue  # this chords was not recognize and we don't want it
                # Use the minimum note_on time as the chord start time
                chords.append((chord_quality, min(chord[:, 1]), chord_map))
            previous_tick = max(onset_notes[:, 1])
            count += len(onset_notes)  # Move to the next notes

        events = []
        for chord in chords:
            events.append(Event(type_='Chord', time=chord[1], value=chord[0], desc=chord[2]))

        if len(events) == 0:
            events.append(Event(type_='Chord', time=0, value=NO_CHORD, desc=NO_CHORD))
        return events

    def detect_chords_by_chorder(self, return_type='events'):
        """Get the better results from 1-beat and 2-beats chord extraction results, implementation is based on
        https://github.com/joshuachang2311/chorder
        NOTE: support detect chord root and chord quality
        """
        chords_1 = get_chords(self.notes, 1, self.time_division)
        chords_2 = get_chords(self.notes, 2, self.time_division)
        chords = []
        for i in range(len(chords_2)):
            prev_index = i * 2
            next_index = i * 2 + 1
            two_chord = chords_2[i]
            two_score = two_chord[1]
            prev_chord = chords_1[prev_index]
            prev_score = prev_chord[1]

            if next_index < len(chords_1):
                next_chord = chords_1[next_index]
                next_score = next_chord[1]

                score_list = [prev_score, next_score, two_score]
                fail_num = Counter(score_list)[-1]

                if fail_num == 0:
                    if (prev_score + next_score) / 2 > two_score:
                        chords += [(prev_chord[0], prev_index * self.time_division),
                                   (next_chord[0], next_index * self.time_division)]
                    else:
                        chords += [(two_chord[0], prev_index * self.time_division),
                                   (two_chord[0], next_index * self.time_division)]

                elif fail_num == 1:
                    if prev_score == -1 or next_score == -1:
                        chords += [(two_chord[0], prev_index * self.time_division),
                                   (two_chord[0], next_index * self.time_division)]
                    else:
                        chords += [(prev_chord[0], prev_index * self.time_division),
                                   (next_chord[0], next_index * self.time_division)]

                elif fail_num == 2:
                    if prev_score != -1:
                        chords += [(prev_chord[0], prev_index * self.time_division),
                                   (NO_CHORD, next_index * self.time_division)]
                    elif next_score != -1:
                        chords += [(NO_CHORD, prev_index * self.time_division),
                                   (next_chord[0], next_index * self.time_division)]
                    else:
                        chords += [(two_chord[0], prev_index * self.time_division),
                                   (two_chord[0], next_index * self.time_division)]

                else:
                    chords += [(NO_CHORD, prev_index * self.time_division),
                               (NO_CHORD, next_index * self.time_division)]

            #     if prev_score != -1 and next_score != -1 and prev_score >= two_score and next_score >= two_score:
            #         chords += [(prev_chord[0], prev_index * self.time_division),
            #                    (next_chord[0], next_index * self.time_division)]
            #     elif (prev_score == -1 or next_score == -1) and two_score != 0:
            #         if prev_score >= two_score:
            #             chords += [(prev_chord[0], prev_index * self.time_division),
            #                        (NO_CHORD, next_index * self.time_division)]
            #         elif next_score >= two_score:
            #             chords += [(NO_CHORD, prev_index * self.time_division),
            #                        (next_chord[0], next_index * self.time_division)]
            #     elif prev_score == -1 and next_score == -1 and two_score == 0:
            #         chords += [(NO_CHORD, prev_index * self.time_division),
            #                    (NO_CHORD, next_index * self.time_division)]
            #     else:
            #         chords += [(two_chord[0], prev_index * self.time_division),
            #                    (two_chord[0], next_index * self.time_division)]
            # else:
            #     if prev_score > two_score:
            #         chords += [(prev_chord[0], prev_index * self.time_division),
            #                    (prev_chord[0], next_index * self.time_division)]
            #         # chords.append((prev_chord[0], prev_index * self.time_division))
            #
            #     else:
            #         chords += [(two_chord[0], prev_index * self.time_division),
            #                    (two_chord[0], next_index * self.time_division)]
            #         # chords.append((two_chord[0], prev_index * self.time_division))
        events = []
        for chord in chords:
            events.append(Event(type_='Chord', time=chord[1], value=chord[0], desc=''))

        if return_type == 'events':
            return events
        elif return_type == 'chords':
            return chords
        else:
            raise AttributeError

    def detect_chords_by_viterbi(self, chords_per_bar=4,
                                 key_change_prob=0.001,
                                 chord_change_prob=0.5,
                                 chord_pitch_out_of_key_prob=0.01,
                                 chord_note_concentration=100.0):
        """Infer chords for a NoteSequence using the Viterbi algorithm.
        Note: use this method may result in some chord results which don't make much sense (but is the most likely),
        if the previous two methods `detect_chords_by_intervals` and `detect_chords_by_chorder` give NO_CHORD results,
        we probably shouldn't trust the results given by this method.

        :param chords_per_bar: If `sequence` is quantized, the number of chords per bar to infer. If None,
         use a default number of chords based on the time signature of `sequence`.
        :param key_change_prob: Probability of a key change between two adjacent frames.
        :param chord_change_prob: Probability of a chord change between two adjacent frames.
        :param chord_pitch_out_of_key_prob: Probability of a pitch in a chord not belonging to the current key.
        :param chord_note_concentration: Concentration parameter for the distribution of observed pitches played
        over a chord. At zero, all pitches are equally likely. As concentration increases, observed pitches must
        match the chord pitches more closely.

        Raises:
          EmptySequenceError: If `sequence` is empty.
          SequenceTooLongError: If the number of chords to be inferred is too large.
        """
        ticks_per_chord = int(self.time_division * 4 / chords_per_bar)
        num_chords = int(self.notes[-1].end / ticks_per_chord)

        if num_chords == 0:
            raise EmptySequenceError('Empty midi')

        if num_chords > MAX_NUM_CHORDS:
            raise SequenceTooLongError(f'Notes too long for chord inference: {num_chords} frames')

        # Compute pitch vectors for each chord frame, then compute log-likelihood of
        # observing those pitch vectors under each possible chord.
        frames_pc_vectors = get_frames_pc_vectors(self.notes, ticks_per_chord)
        # At each time step(frame), the similarity score of notes between different chords.
        frame_chord_loglik = get_chord_frame_log_likelihood(frames_pc_vectors, chord_note_concentration)
        # Compute distribution over chords for each key, and transition distribution between key-chord pairs.
        key_chord_distribution = get_key_chord_distribution(chord_pitch_out_of_key_prob)
        key_chord_transition_distribution = get_key_chord_transition_distribution(key_chord_distribution,
                                                                                  key_change_prob,
                                                                                  chord_change_prob)
        key_chord_loglik = np.log(key_chord_distribution)
        key_chord_transition_loglik = np.log(key_chord_transition_distribution)

        key_chords = perform_key_chord_viterbi(frame_chord_loglik, key_chord_loglik, key_chord_transition_loglik)
        # Add the inferred chord changes to the sequence
        chords = []
        for frame, (key, chord) in enumerate(key_chords):
            if chord == NO_CHORD:
                figure = NO_CHORD
            else:
                root, kind = chord
                figure = '%s_%s' % (root, kind)
            chords.append((figure, frame * ticks_per_chord))

        # Deduplication
        events = []
        prev_chord = None
        notes_on_set = set()
        for n in self.notes:
            if n.start not in notes_on_set:
                notes_on_set.add(n.start)
        for chord in chords:
            if chord[0] != prev_chord and chord[1] in notes_on_set:
                events.append(Event(type_='Chord', time=chord[1], value=chord[0].split("_")[-1], desc=''))
                prev_chord = chord[0]
        return events
