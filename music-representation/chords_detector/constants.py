# -*- coding: utf-8 -*-
""" Constants for data encoding

"""
import itertools
import numpy as np

# 将条件输入和弦转换为标准输入格式
NUMBER2CHORDS = {"0": "0_maj7",
                 "2": "2_m7",
                 "4": "4_m7",
                 "5": "5_maj7",
                 "7": "7_7",
                 "9": "9_m7",
                 "11": "11_m7b5",
                 "0_": "0_",
                 "2_": "2_m",
                 "4_": "4_m",
                 "5_": "5_",
                 "7_": "7_",
                 "9_": "9_m",
                 "11_": "11_dim",
                 }

# 自然音阶(scale)，对应5个全音和2个半音的组合，组成7种调式: Lydian, Ionian, Mixolydian, Dorian, Aeolian, Phrygian and Locrian
DIATONIC_PITCHES = [0, 2, 4, 5, 7, 9, 11]

# 12 pitch classes
PITCH_CLASS_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

NOTE2PC = {'C': 0,
           'C#': 1,
           'D': 2,
           'Eb': 3,
           'E': 4,
           'F': 5,
           'F#': 6,
           'G': 7,
           'Ab': 8,
           'A': 9,
           'Bb': 10,
           'B': 11}

PC2NOTE = {0: 'C',
           1: 'C#',
           2: 'D',
           3: 'Eb',
           4: 'E',
           5: 'F',
           6: 'F#',
           7: 'G',
           8: 'Ab',
           9: 'A',
           10: 'Bb',
           11: 'B'}

# Chord symbol for "no chord".
NO_CHORD = 'N.C.'

# Mapping from time signature to number of chords to infer per bar.
DEFAULT_TIME_SIGNATURE_CHORDS_PER_BAR = {
    (2, 2): 1,
    (2, 4): 1,
    (3, 4): 1,
    (4, 4): 4,
    (6, 8): 2,
}

# Maximum length of chord sequence to infer.
MAX_NUM_CHORDS = 10000

# 自然音阶下的7种调式(modes)
DIATONIC_MODES = [
    # Ionian（伊奥尼亚调式） 也称为自然大调，自然大调的I级音阶，由C大调的C进行到高八度的C，构成音分别为：1 2 3 4 5 6 7 1
    [0, 2, 4, 5, 7, 9, 11],  # 全全半全全全半
    # Locrian（洛克里亚调式） 自然大调的VII级音阶，由C大调的B进行到高八度的B，构成音分别为：1 ♭2 ♭3 4 ♭5 ♭6 ♭7 1
    [0, 1, 3, 5, 6, 8, 10],  # 半全全半全全全
    [1, 2, 4, 6, 7, 9, 11],  # 半全全半全全全
    # Aeolian（爱奥尼亚调式） 也称为自然小调，自然大调的VI级音阶，由C大调的A进行到高八度的A，构成音分别为：1 2 ♭3 4 5 ♭6 ♭7 1
    [0, 2, 3, 5, 7, 8, 10],  # 全半全全半全全
    [1, 3, 4, 6, 8, 9, 11],  # 全半全全半全全
    # Mixo-lydian（混合利底亚调式） 自然大调的V级音阶，由C大调的G进行到高八度的G，构成音分别为：1 2 3 4 5 6 ♭7 1
    [0, 2, 4, 5, 7, 9, 10],  # 全全半全全半; 大调的3音，5音，7音降半音
    [1, 3, 5, 6, 8, 10, 11],
    # Lydian（利底亚调式） 自然大调的 IV 级音阶，由 C 大调的 F 进行到高八度的 F，构成音分别为：1 2 3 ♯4 5 6 7 1
    [0, 2, 4, 6, 7, 9, 11],  # 全全全半全全半
    # Phrygian（弗里几亚调式） 自然大调的 III 级音阶，由 C 大调的 E 进行到高八度的 E，构成音分别为：1 ♭2 ♭3 4 5 ♭6 ♭7 1
    [0, 1, 3, 5, 7, 8, 10],  # 半全全全半全全
    [1, 2, 4, 6, 8, 9, 11],
    # Dorian（多利亚调式） 自然大调的 II 级音阶，由 C 大调的 D 进行到高八度的 D，构成音分别为：1 2 ♭3 4 5 6 ♭7 1
    [0, 2, 3, 5, 7, 9, 10],  # 全半全全全半全
    [1, 3, 4, 6, 8, 10, 11]
]

# Simplified chord kinds used in google magenta
MAGENTA_CHORD_KIND_PITCHES = {
    '': [0, 4, 7],
    'm': [0, 3, 7],
    '+': [0, 4, 8],
    'dim': [0, 3, 6],
    '7': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    'm7b5': [0, 3, 6, 10],
}

# 常见和弦音程(interval)关系, 目前使用14种
CHORD_MAPS = {'': (0, 4, 7),  # maj, major, 大三度 + 小三度 = 大三和弦
              'm': (0, 3, 7),  # min, minor, 小三度 + 大三度 = 小三和弦
              '+': (0, 4, 8),  # aug, augmented, 大三 + 大三 = 增三和弦, also notated as '+'
              'dim': (0, 3, 6),  # dim, diminished, 小三 + 小三 = 减三和弦, also notated as 'o'
              '7': (0, 4, 7, 10),  # 7dom, dominant, 大三+小三+小三（小7度， 10-0），属7和弦
              'maj7': (0, 4, 7, 11),  # 7maj, major seventh, 大三+小三+大三（大7度，11-0），大7和弦
              'm7': (0, 3, 7, 10),  # 7min, minor seventh, 小三+大三+小三， 小7和弦
              'm7b5': (0, 3, 6, 10),  # 7halfdim, half diminished seventh, 小三+小三+大三（小7度，10-0），半减7和弦（小7减5或小7降5和弦m7b5）
              # 'sus2': (0, 2, 7),  # suspend-2, 和弦三音换成二音， 大二度 + 四度，cmajor-萨斯2 or c-萨斯2
              # 'sus4': (0, 5, 7),  # suspend-4, 和弦三音换成四音， 四度 + 大二度
              # '7dim': (0, 3, 6, 9),  # diminished seventh, 小三+小三+小三（大6度，9-0），减7和弦
              # '7aug': (0, 4, 8, 10),  # 大三+大三+大二（小7度，10-0），增7和弦（增小7和弦或增属7和弦）
              # '7augM7': (0, 4, 8, 11),  # 大三+大三+小三（大7度， 11-0）， 增大7和弦, extra
              # '7mM7': (0, 3, 7, 11),  # 小三+大三+大三（大7度， 11-0）， 小大7和弦, extra
              # '9maj': (0, 4, 7, 11, 14),  # 大三+小三+大三+小三（大9度，14-0），大9和弦
              # '9min': (0, 3, 7, 10, 14),  # 小三+大三+小三+大三（大9度，14-0），小9和弦
              }

CHORD_WEIGHTS = {'': [10, -2, -1, -2, 10, -5, -2, 10, -2, -1, -2, 0],
                 'm': [10, -2, -1, 10, -2, -5, -2, 10, -1, -2, 0, -2],
                 '+': [10, -2, -1, -2, 10, -1, -2, -2, 10, -1, -2, 0],
                 'dim': [10, -2, -1, 10, -2, -1, 10, -2, -2, 1, -1, -2],
                 '7': [10, -2, -1, -2, 10, -5, -2, 10, -2, -1, 10, 0],
                 'maj7': [10, -2, -1, -2, 10, -5, -2, 10, -2, -1, -1, 10],
                 'm7': [10, -2, -1, 10, -2, -5, -2, 10, -1, -2, 10, -2],
                 'm7b5': [10, -2, -1, 10, -2, -1, 10, -2, -2, -1, 10, -2],
                 # 'sus2': [10, -2, 5, -2, -1, -5, -2, 5, -2, -1, -2, 0],
                 # 'sus4': [10, -2, -1, -2, -5, 5, -2, 5, -2, -1, -2, 0],
                 # '7dim': [10, -2, -1, 10, -2, -1, 10, -2, -2, 10, -1, -2],
                 # '7aug': [10, -2, -1, -2, 10, -1, -2, -2, 10, -1, 10, 0],
                 # '7augM7': [10, -2, -1, -2, 10, -1, -2, -2, 10, -1, -1, 10],
                 # '7mM7': [10, -2, -1, 10, -2, -5, -2, 10, -1, -2, -2, 10],
                 }

# All chord kinds (chord qualities)
CHORD_KINDS = CHORD_MAPS.keys()
# TIP: too many chord kinds may result in the slowness of computation, if so, consider using fewer chord kinds as below
MAIN_CHORD_KINDS = ['maj', 'min', 'dim', 'aug', '7dom', '7maj', '7min', '7halfdim']

# All usable chords, including no-chord.
ALL_CHORDS = [NO_CHORD] + list(itertools.product(range(12), CHORD_KINDS))
# All key-chord pairs.
ALL_KEY_CHORDS = list(itertools.product(range(12), ALL_CHORDS))


def get_chord_pitch_vectors():
    """Unit vectors over pitch classes for all chords."""
    x = np.zeros([len(ALL_CHORDS), 12])
    for i, chord in enumerate(ALL_CHORDS[1:]):
        root, kind = chord
        for offset in CHORD_MAPS[kind]:
            x[i + 1, (root + offset) % 12] = 1
    x[1:, :] /= np.linalg.norm(x[1:, :], axis=1)[:, np.newaxis]
    return x


CHORD_PITCH_VECTORS = get_chord_pitch_vectors()

# Default MIDI encodings parameters
# PITCH_RANGE = range(21, 109)  # the recommended pitches for piano in the GM2 specs are from 21 to 108
PITCH_RANGE = range(128)
# BEAT_RES = {(0, 8): 4}  # duration's maximum is 8 beats(2 bars) and we have 4 samples per beat
BEAT_RES = {(0, 8): 8}  # duration's maximum is 8 beats(2 bars) and we have 4 samples per beat
NB_VELOCITIES = 32  # nb of velocity bins, velocities values from 0 to 127 will be quantized
ADDITIONAL_TOKENS = {'Chord': True,
                     'Rest': False,
                     'Tempo': False,
                     'TimeSignature': False,
                     'Program': False,
                     # rest params, default value: minimum 1/2 beats, maximum 8 beats
                     'rest_range': (2, 8),  # (/min_rest, max_rest_in_BEAT), first divides a whole note/rest
                     # tempo params
                     'nb_tempos': 32,  # nb of tempo bins for additional tempo tokens, quantized like velocities
                     'tempo_range': (40, 250),  # (min_tempo, max_tempo)
                     # time signature params, maximum 8th note
                     'time_signature_range': (8, 2)}  # (max_beat_res, max_bar_length_in_NOTE)
INSTRUMENT2TRACK = {'Drums': 1, 'Piano': 2, 'Guitar': 3, 'Bass': 4, 'Strings': 5, 'Lead': 6}
TRACK_NAMES = {
    (0, True): 'Drums',
    (0, False): 'Piano',
    (25, False): 'Guitar',
    (32, False): 'Bass',
    (48, False): 'Strings',
    (80, False): 'Lead',
}
# latest version of instrument classification
INSTR_DICT = {
    'PIANO': list(range(0, 8)),  # P1
    'CHROMATIC PERCUSSION': list(range(8, 16)),
    'ORGAN': list(range(16, 24)),
    'GUITAR': list(range(24, 32)),  # P2
    'BASS': list(range(32, 40)),  # P3
    'STRING': list(range(40, 53)),  # P4
    'ENSEMBLE': list(range(53, 56)),
    'BRASS': list(range(56, 64)),
    'REED': list(range(64, 72)),
    'PIPE': list(range(72, 80)),
    'SYNTH LEAD': list(range(80, 88)),
    'SYNTH PAD': list(range(88, 96)),
    'SYNTH EFFECTS': list(range(96, 104)),
    'ETHNIC': list(range(104, 112)),
    'PERCUSSIVE': list(range(112, 120)),
    'DRUM': list(range(112, 120)),  # P5
    'SOUND EFFECTS': list(range(120, 128)),
}


# The accumulated number of different instruments in the training data. EFFECT is more like functional, so it shouldn't
# be included as instruments
INSTR_DICT_NUM = {'GUITAR': 91749, 'DRUM': 73334, 'STRING': 68216, 'BASS': 63672, 'PIANO': 60595, 'EFFECT': 42785,
                  'BRASS': 39263, 'REED': 27516, 'PIPE': 18805, 'SYNTH LEAD': 17793, 'ORGAN': 17646, 'ENSEMBLE': 12697,
                  'SYNTH PAD': 11965, 'CHROMATIC PERCUSSION': 9730, 'SYNTH EFFECTS': 4950, 'ETHNIC': 4647,
                  'PERCUSSIVE': 3049, 'SOUND EFFECTS': 1287}

MUSIC_FUNCTIONS = ['MAIN MELODY', 'SUB MELODY', 'ACCOMPANIMENT']
INDEX2TRACKS = list(itertools.product(INSTR_DICT.keys(), MUSIC_FUNCTIONS))
TRACKS2INDEX = {v: k for k, v in enumerate(INDEX2TRACKS)}

DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]
# Defaults values when writing new MIDI files
# 384 and 480 are convenient as divisible by 4, 8, 12, 16, 24, 32, by the way, they are commonly used in modern DAW
TIME_DIVISION = 384  # or 480
TEMPO = 120
TIME_SIGNATURE = (4, 4)

# http://newt.phys.unsw.edu.au/jw/notes.html
# https://www.midi.org/specifications

# index i = program i+1 in the GM2 specs (7. Appendix A)
# index i = program i as retrieved by packages like mido or miditoolkit
MIDI_INSTRUMENTS = [{'name': 'Acoustic Grand Piano', 'pitch_range': (21, 108)},
                    {'name': 'Bright Acoustic Piano', 'pitch_range': (21, 108)},
                    {'name': 'Electric Grand Piano', 'pitch_range': (21, 108)},
                    {'name': 'Honky-tonk Piano', 'pitch_range': (21, 108)},
                    {'name': 'Electric Piano 1', 'pitch_range': (28, 103)},
                    {'name': 'Electric Piano 2', 'pitch_range': (28, 103)},
                    {'name': 'Harpsichord', 'pitch_range': (41, 89)},
                    {'name': 'Clavi', 'pitch_range': (36, 96)},

                    # Chromatic Percussion
                    {'name': 'Celesta', 'pitch_range': (60, 108)},
                    {'name': 'Glockenspiel', 'pitch_range': (72, 108)},
                    {'name': 'Music Box', 'pitch_range': (60, 84)},
                    {'name': 'Vibraphone', 'pitch_range': (53, 89)},
                    {'name': 'Marimba', 'pitch_range': (48, 84)},
                    {'name': 'Xylophone', 'pitch_range': (65, 96)},
                    {'name': 'Tubular Bells', 'pitch_range': (60, 77)},
                    {'name': 'Dulcimer', 'pitch_range': (60, 84)},

                    # Organs
                    {'name': 'Drawbar Organ', 'pitch_range': (36, 96)},
                    {'name': 'Percussive Organ', 'pitch_range': (36, 96)},
                    {'name': 'Rock Organ', 'pitch_range': (36, 96)},
                    {'name': 'Church Organ', 'pitch_range': (21, 108)},
                    {'name': 'Reed Organ', 'pitch_range': (36, 96)},
                    {'name': 'Accordion', 'pitch_range': (53, 89)},
                    {'name': 'Harmonica', 'pitch_range': (60, 84)},
                    {'name': 'Tango Accordion', 'pitch_range': (53, 89)},

                    # Guitars
                    {'name': 'Acoustic Guitar (nylon)', 'pitch_range': (40, 84)},
                    {'name': 'Acoustic Guitar (steel)', 'pitch_range': (40, 84)},
                    {'name': 'Electric Guitar (jazz)', 'pitch_range': (40, 86)},
                    {'name': 'Electric Guitar (clean)', 'pitch_range': (40, 86)},
                    {'name': 'Electric Guitar (muted)', 'pitch_range': (40, 86)},
                    {'name': 'Overdriven Guitar', 'pitch_range': (40, 86)},
                    {'name': 'Distortion Guitar', 'pitch_range': (40, 86)},
                    {'name': 'Guitar Harmonics', 'pitch_range': (40, 86)},

                    # Bass
                    {'name': 'Acoustic Bass', 'pitch_range': (28, 55)},
                    {'name': 'Electric Bass (finger)', 'pitch_range': (28, 55)},
                    {'name': 'Electric Bass (pick)', 'pitch_range': (28, 55)},
                    {'name': 'Fretless Bass', 'pitch_range': (28, 55)},
                    {'name': 'Slap Bass 1', 'pitch_range': (28, 55)},
                    {'name': 'Slap Bass 2', 'pitch_range': (28, 55)},
                    {'name': 'Synth Bass 1', 'pitch_range': (28, 55)},
                    {'name': 'Synth Bass 2', 'pitch_range': (28, 55)},

                    # Strings & Orchestral instruments
                    {'name': 'Violin', 'pitch_range': (55, 93)},
                    {'name': 'Viola', 'pitch_range': (48, 84)},
                    {'name': 'Cello', 'pitch_range': (36, 72)},
                    {'name': 'Contrabass', 'pitch_range': (28, 55)},
                    {'name': 'Tremolo Strings', 'pitch_range': (28, 93)},
                    {'name': 'Pizzicato Strings', 'pitch_range': (28, 93)},
                    {'name': 'Orchestral Harp', 'pitch_range': (23, 103)},
                    {'name': 'Timpani', 'pitch_range': (36, 57)},

                    # Ensembles
                    {'name': 'String Ensembles 1', 'pitch_range': (28, 96)},
                    {'name': 'String Ensembles 2', 'pitch_range': (28, 96)},
                    {'name': 'SynthStrings 1', 'pitch_range': (36, 96)},
                    {'name': 'SynthStrings 2', 'pitch_range': (36, 96)},
                    {'name': 'Choir Aahs', 'pitch_range': (48, 79)},
                    {'name': 'Voice Oohs', 'pitch_range': (48, 79)},
                    {'name': 'Synth Voice', 'pitch_range': (48, 84)},
                    {'name': 'Orchestra Hit', 'pitch_range': (48, 72)},

                    # Brass
                    {'name': 'Trumpet', 'pitch_range': (58, 94)},
                    {'name': 'Trombone', 'pitch_range': (34, 75)},
                    {'name': 'Tuba', 'pitch_range': (29, 55)},
                    {'name': 'Muted Trumpet', 'pitch_range': (58, 82)},
                    {'name': 'French Horn', 'pitch_range': (41, 77)},
                    {'name': 'Brass Section', 'pitch_range': (36, 96)},
                    {'name': 'Synth Brass 1', 'pitch_range': (36, 96)},
                    {'name': 'Synth Brass 2', 'pitch_range': (36, 96)},

                    # Reed
                    {'name': 'Soprano Sax', 'pitch_range': (54, 87)},
                    {'name': 'Alto Sax', 'pitch_range': (49, 80)},
                    {'name': 'Tenor Sax', 'pitch_range': (42, 75)},
                    {'name': 'Baritone Sax', 'pitch_range': (37, 68)},
                    {'name': 'Oboe', 'pitch_range': (58, 91)},
                    {'name': 'English Horn', 'pitch_range': (52, 81)},
                    {'name': 'Bassoon', 'pitch_range': (34, 72)},
                    {'name': 'Clarinet', 'pitch_range': (50, 91)},

                    # Pipe
                    {'name': 'Piccolo', 'pitch_range': (74, 108)},
                    {'name': 'Flute', 'pitch_range': (60, 96)},
                    {'name': 'Recorder', 'pitch_range': (60, 96)},
                    {'name': 'Pan Flute', 'pitch_range': (60, 96)},
                    {'name': 'Blown Bottle', 'pitch_range': (60, 96)},
                    {'name': 'Shakuhachi', 'pitch_range': (55, 84)},
                    {'name': 'Whistle', 'pitch_range': (60, 96)},
                    {'name': 'Ocarina', 'pitch_range': (60, 84)},

                    # Synth Lead
                    {'name': 'Lead 1 (square)', 'pitch_range': (21, 108)},
                    {'name': 'Lead 2 (sawtooth)', 'pitch_range': (21, 108)},
                    {'name': 'Lead 3 (calliope)', 'pitch_range': (36, 96)},
                    {'name': 'Lead 4 (chiff)', 'pitch_range': (36, 96)},
                    {'name': 'Lead 5 (charang)', 'pitch_range': (36, 96)},
                    {'name': 'Lead 6 (voice)', 'pitch_range': (36, 96)},
                    {'name': 'Lead 7 (fifths)', 'pitch_range': (36, 96)},
                    {'name': 'Lead 8 (bass + lead)', 'pitch_range': (21, 108)},

                    # Synth Pad
                    {'name': 'Pad 1 (new age)', 'pitch_range': (36, 96)},
                    {'name': 'Pad 2 (warm)', 'pitch_range': (36, 96)},
                    {'name': 'Pad 3 (polysynth)', 'pitch_range': (36, 96)},
                    {'name': 'Pad 4 (choir)', 'pitch_range': (36, 96)},
                    {'name': 'Pad 5 (bowed)', 'pitch_range': (36, 96)},
                    {'name': 'Pad 6 (metallic)', 'pitch_range': (36, 96)},
                    {'name': 'Pad 7 (halo)', 'pitch_range': (36, 96)},
                    {'name': 'Pad 8 (sweep)', 'pitch_range': (36, 96)},

                    # Synth SFX
                    {'name': 'FX 1 (rain)', 'pitch_range': (36, 96)},
                    {'name': 'FX 2 (soundtrack)', 'pitch_range': (36, 96)},
                    {'name': 'FX 3 (crystal)', 'pitch_range': (36, 96)},
                    {'name': 'FX 4 (atmosphere)', 'pitch_range': (36, 96)},
                    {'name': 'FX 5 (brightness)', 'pitch_range': (36, 96)},
                    {'name': 'FX 6 (goblins)', 'pitch_range': (36, 96)},
                    {'name': 'FX 7 (echoes)', 'pitch_range': (36, 96)},
                    {'name': 'FX 8 (sci-fi)', 'pitch_range': (36, 96)},

                    # Ethnic Misc.
                    {'name': 'Sitar', 'pitch_range': (48, 77)},
                    {'name': 'Banjo', 'pitch_range': (48, 84)},
                    {'name': 'Shamisen', 'pitch_range': (50, 79)},
                    {'name': 'Koto', 'pitch_range': (55, 84)},
                    {'name': 'Kalimba', 'pitch_range': (48, 79)},
                    {'name': 'Bag pipe', 'pitch_range': (36, 77)},
                    {'name': 'Fiddle', 'pitch_range': (55, 96)},
                    {'name': 'Shanai', 'pitch_range': (48, 72)},

                    # Percussive
                    {'name': 'Tinkle Bell', 'pitch_range': (72, 84)},
                    {'name': 'Agogo', 'pitch_range': (60, 72)},
                    {'name': 'Steel Drums', 'pitch_range': (52, 76)},
                    {'name': 'Woodblock', 'pitch_range': (0, 127)},
                    {'name': 'Taiko Drum', 'pitch_range': (0, 127)},
                    {'name': 'Melodic Tom', 'pitch_range': (0, 127)},
                    {'name': 'Synth Drum', 'pitch_range': (0, 127)},
                    {'name': 'Reverse Cymbal', 'pitch_range': (0, 127)},

                    # SFX
                    {'name': 'Guitar Fret Noise, Guitar Cutting Noise', 'pitch_range': (0, 127)},
                    {'name': 'Breath Noise, Flute Key Click', 'pitch_range': (0, 127)},
                    {'name': 'Seashore, Rain, Thunder, Wind, Stream, Bubbles', 'pitch_range': (0, 127)},
                    {'name': 'Bird Tweet, Dog, Horse Gallop', 'pitch_range': (0, 127)},
                    {'name': 'Telephone Ring, Door Creaking, Door, Scratch, Wind Chime', 'pitch_range': (0, 127)},
                    {'name': 'Helicopter, Car Sounds', 'pitch_range': (0, 127)},
                    {'name': 'Applause, Laughing, Screaming, Punch, Heart Beat, Footstep', 'pitch_range': (0, 127)},
                    {'name': 'Gunshot, Machine Gun, Lasergun, Explosion', 'pitch_range': (0, 127)}]

INSTRUMENT_CLASSES = dict([(n, (0, 'Piano')) for n in range(0, 8)] +
                          [(n, (1, 'Chromatic Percussion')) for n in range(8, 16)] +
                          [(n, (2, 'Organ')) for n in range(16, 24)] +
                          [(n, (3, 'Guitar')) for n in range(24, 32)] +
                          [(n, (4, 'Bass')) for n in range(32, 40)] +
                          [(n, (5, 'Strings')) for n in range(40, 48)] +
                          [(n, (6, 'Ensemble')) for n in range(48, 56)] +
                          [(n, (7, 'Brass')) for n in range(56, 64)] +
                          [(n, (8, 'Reed')) for n in range(64, 72)] +
                          [(n, (9, 'Pipe')) for n in range(72, 80)] +
                          [(n, (10, 'Synth Lead')) for n in range(80, 88)] +
                          [(n, (11, 'Synth Pad')) for n in range(88, 96)] +
                          [(n, (12, 'Synth Effects')) for n in range(96, 104)] +
                          [(n, (13, 'Ethnic')) for n in range(104, 112)] +
                          [(n, (14, 'Percussive')) for n in range(112, 120)] +
                          [(n, (15, 'Sound Effects')) for n in range(120, 128)] +
                          [(-1, (-1, 'Drums'))])

INSTRUMENT_CLASSES_RANGES = {'Piano': (0, 7), 'Chromatic Percussion': (8, 15), 'Organ': (16, 23), 'Guitar': (24, 31),
                             'Bass': (32, 39), 'Strings': (40, 47), 'Ensemble': (48, 55), 'Brass': (56, 63),
                             'Reed': (64, 71),
                             'Pipe': (72, 79), 'Synth Lead': (80, 87), 'Synth Pad': (88, 95),
                             'Synth Effects': (96, 103),
                             'Ethnic': (104, 111), 'Percussive': (112, 119), 'Sound Effects': (120, 127), 'Drums': -1}

# index i = program i+1 in the GM2 specs (8. Appendix B)
# index i = program i as retrieved by packages like mido or miditoolkit
DRUM_SETS = {0: 'Standard', 8: 'Room', 16: 'Power', 24: 'Electronic', 25: 'Analog', 32: 'Jazz', 40: 'Brush',
             48: 'Orchestra', 56: 'SFX'}

# Control changes list (without specifications):
# https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2
# Undefined and general control changes are not considered here
# All these attributes can take values from 0 to 127, with some of them being on/off
CONTROL_CHANGES = {
    # MSB
    0: 'Bank Select',
    1: 'Modulation Depth',
    2: 'Breath Controller',
    4: 'Foot Controller',
    5: 'Portamento Time',
    6: 'Data Entry',
    7: 'Channel Volume',
    8: 'Balance',
    10: 'Pan',
    11: 'Expression Controller',

    # LSB
    32: 'Bank Select',
    33: 'Modulation Depth',
    34: 'Breath Controller',
    36: 'Foot Controller',
    37: 'Portamento Time',
    38: 'Data Entry',
    39: 'Channel Volume',
    40: 'Balance',
    42: 'Pan',
    43: 'Expression Controller',

    # On / Off control changes, ≤63 off, ≥64 on
    64: 'Damper Pedal',
    65: 'Portamento',
    66: 'Sostenuto',
    67: 'Soft Pedal',
    68: 'Legato Footswitch',
    69: 'Hold 2',

    # Continuous controls
    70: 'Sound Variation',
    71: 'Timbre/Harmonic Intensity',
    72: 'Release Time',
    73: 'Attack Time',
    74: 'Brightness',
    75: 'Decay Time',
    76: 'Vibrato Rate',
    77: 'Vibrato Depth',
    78: 'Vibrato Delay',
    84: 'Portamento Control',
    88: 'High Resolution Velocity Prefix',

    # Effects depths
    91: 'Reverb Depth',
    92: 'Tremolo Depth',
    93: 'Chorus Depth',
    94: 'Celeste Depth',
    95: 'Phaser Depth',

    # Registered parameters numbers
    96: 'Data Increment',
    97: 'Data Decrement',
    #  98: 'Non-Registered Parameter Number (NRPN) - LSB',
    #  99: 'Non-Registered Parameter Number (NRPN) - MSB',
    100: 'Registered Parameter Number (RPN) - LSB',
    101: 'Registered Parameter Number (RPN) - MSB',

    # Channel mode controls
    120: 'All Sound Off',
    121: 'Reset All Controllers',
    122: 'Local Control On/Off',
    123: 'All Notes Off',
    124: 'Omni Mode Off',  # + all notes off
    125: 'Omni Mode On',  # + all notes off
    126: 'Mono Mode On',  # + poly off, + all notes off
    127: 'Poly Mode On'  # + mono off, +all notes off
}
