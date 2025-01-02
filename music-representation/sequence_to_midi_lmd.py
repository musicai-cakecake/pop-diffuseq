import copy
import json

pit2alphabet = ['C', 'd', 'D', 'e', 'E', 'F', 'g', 'G', 'a', 'A', 'b', 'B']
char2pit = {x: id for id, x in enumerate(pit2alphabet)}
from miditoolkit.midi.containers import Note as mtkNote
from miditoolkit.midi.parser import MidiFile
from miditoolkit.midi.containers import Instrument
from miditoolkit.midi.containers import TempoChange

data_sequence = []

instrument_to_program_dict = {
    '0_MELODY': int(0),
    'PIANO_ACCOMPANIMENT': int(0),  # 钢琴 0-7
    'CHROMATIC_PERCUSSION_ACCOMPANIMENT': int(12),  # 色彩打击乐器 8-15 比如12马林巴
    'ORGAN_ACCOMPANIMENT': int(21),  # 风琴 16-23
    'ACOUSTIC_GUITAR_ACCOMPANIMENT': int(25),  # 吉他 24-31，25民谣，29电吉他
    'ELECTRIC_GUITAR_ACCOMPANIMENT': int(29),
    'BASS_ACCOMPANIMENT': int(32),  # 贝司 32-39 33指弹贝司 32 录音室贝司
    'STRING_ACCOMPANIMENT': int(41),  # 弦乐独奏 40-47  # 并入弦乐合奏
    'STRINGS_ACCOMPANIMENT': int(48),  # 弦乐合奏 48-51
    'ENSEMBLE_ACCOMPANIMENT': int(51),  # 合奏，合唱 48-55
    'BRASS_ACCOMPANIMENT': int(61),  # 铜管 56-63
    'REED_ACCOMPANIMENT': int(70),  # 木管，簧管类 64-71
    'PIPE_ACCOMPANIMENT': int(73),  # 笛子 72-79
    'SYNTH_LEAD_ACCOMPANIMENT': int(80),  # 合成主音 80-87
    'SYNTH_PAD_ACCOMPANIMENT': int(90),  # 合成音色 88-95
    'SYNTH_EFFECTS_ACCOMPANIMENT': int(100),  # 合成效果 96-103
    'ETHNIC_ACCOMPANIMENT': int(104),  # 民族 104-111
    'PERCUSSIVE_ACCOMPANIMENT': int(118),  # 打击乐器 112-119， 116 管弦乐套件，118 鼓合成套件

    'SOUND_EFFECTS_ACCOMPANIMENT': int(122),  # 声音效果 120-127  # 不要

    'DRUM_ACCOMPANIMENT': int(0),  # 鼓

    'UNKNOWN': int(0)

}

# instrument_category_and_program_dict = {
#         'PIANO_ACCOMPANIMENT': [0, 7],  # 钢琴
#         'CHROMATIC_PERCUSSION_ACCOMPANIMENT': [8, 15],  # 色彩打击乐器，is_drum = False
#         'ORGAN_ACCOMPANIMENT': [16, 23],  # 风琴
#         'ACOUSTIC_GUITAR_ACCOMPANIMENT': [24, 25],  # 声学吉他
#         'ELECTRIC_GUITAR_ACCOMPANIMENT': [26, 32],  # 电音吉他
#         'BASS_ACCOMPANIMENT': [32, 39],  # 贝司
#         'STRINGS_ACCOMPANIMENT': [40, 47],  # 弦乐
#         'ENSEMBLE_ACCOMPANIMENT': [48, 55],  # 合唱
#         'BRASS_ACCOMPANIMENT': [56, 63],  # 铜管
#         'REED_ACCOMPANIMENT': [64, 71],  # 簧管
#         'PIPE_ACCOMPANIMENT': [72, 79],  # 笛
#         'SYNTH_LEAD_ACCOMPANIMENT': [80, 87],  # 合成主音
#         'SYNTH_PAD_ACCOMPANIMENT': [88, 95],  # 合成铺垫
#         'SYNTH_EFFECTS_ACCOMPANIMENT': [96, 103],  # 合成效果
#         'ETHNIC_ACCOMPANIMENT': [104, 111],  # 民族乐器
#         'PERCUSSIVE_ACCOMPANIMENT': [112, 119],  # 打击乐器
#         'SOUND_EFFECTS_ACCOMPANIMENT': [120, 127]  # 声音效果
#     }

tempo_to_bpm_dict = {
    'Tempo_Largo': float(60.0),
    'Tempo_Andante': float(96.0),  # 钢琴 0-7
    'Tempo_Moderato': float(120.0),  # 色彩打击乐器 8-15 比如马林巴，钢片琴
    'Tempo_Allegro': float(138.0),  # 风琴 16-23
    'Tempo_Presto': float(144.0),  # 吉他 24-31
}


def str2pit(x):
    rel_pit = char2pit[x[0]]
    octave = (int(x[1]) if x[1] != 'O' else -1) + 1
    return octave * 12 + rel_pit


def note_seq_to_track(note_seq, ticks_per_beat=480):

    tickes_per_32th = ticks_per_beat // 8
    # tickes_per_16th = ticks_per_beat // 4

    tracks = {}
    for pitch, program, is_drum, start, end, track_name in note_seq:
        if pitch != 'None':
            tracks.setdefault((program, is_drum, track_name), []).append(
                mtkNote(90, pitch, start * tickes_per_32th, end * tickes_per_32th)
            )

    instruments = []

    for tp, notes in tracks.items():
        instrument = Instrument(program=int(tp[0]), is_drum=tp[1], name=str(tp[2]))

        # instrument.program = tp[1] % 128
        # instrument.is_drum = tp[1] > 128
        instrument.notes = notes
        instrument.remove_notes_with_no_duration()
        instruments.append(copy.deepcopy(instrument))

    return instruments


def tokens_to_note_event_seq(tokens_str: str, chord_absolute_time=int(0)):

    music_event = tokens_str.split(" ")

    note_sequence = []

    instrument_name_list = ['0_MELODY', 'ENSEMBLE_ACCOMPANIMENT', 'SYNTH_LEAD_ACCOMPANIMENT', 'BRASS_ACCOMPANIMENT',
                            'ACOUSTIC_GUITAR_ACCOMPANIMENT', 'ELECTRIC_GUITAR_ACCOMPANIMENT', 'ORGAN_ACCOMPANIMENT',
                            'BASS_ACCOMPANIMENT', 'DRUM_ACCOMPANIMENT',
                            'ETHNIC_ACCOMPANIMENT', 'PIANO_ACCOMPANIMENT', 'REED_ACCOMPANIMENT',
                            'STRINGS_ACCOMPANIMENT',
                            'PIPE_ACCOMPANIMENT', 'SOUND_EFFECTS_ACCOMPANIMENT', 'CHROMATIC_PERCUSSION_ACCOMPANIMENT',
                            'SYNTH_PAD_ACCOMPANIMENT', 'STRING_ACCOMPANIMENT', 'PERCUSSIVE_ACCOMPANIMENT',
                            'SYNTH_EFFECTS_ACCOMPANIMENT', 'PIANO_COMPOUND', 'REED_COMPOUND']

    # track_number = int(0)
    # track_dict = {}

    # 计算各种音符的绝对时间，并封装成序列的结构
    for i in range(len(music_event) - 3):

        if music_event[i] == '0_CHORD' and music_event[i+1][:4] == 'time' and music_event[i+2][:5] == 'chord':
            # if isinstance(music_event[i + 1][4:], int):
            chord_absolute_time += int(music_event[i+1][4:])

        elif music_event[i] in instrument_name_list:
            # if music_event[i] not in track_dict.keys():
            #     track_dict[music_event[i]] = track_number
            #     track_number += 1

            if music_event[i] in ['DRUM_ACCOMPANIMENT'] or music_event[i] in ['PERCUSSIVE_ACCOMPANIMENT']:

                is_drum = True
            else:
                is_drum = False

            # note_track = track_dict[music_event[i]]
            note_relative_time = music_event[i + 1]
            note_pitch = music_event[i + 2]
            note_duration = music_event[i + 3]

            if 'time' in note_relative_time and 'dur' in note_duration and 'chord' not in note_pitch:
                if note_pitch[0] in pit2alphabet and \
                        note_pitch[1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'] and 'dur' not in note_pitch:
                    # 音符事件
                    current_time = int(note_relative_time[4:]) + chord_absolute_time
                    pitch = str2pit(note_pitch)
                    # duration 是16分音符为单位，但是时间是32分音符为单位
                    duration = int(note_duration[3:]) * 2
                    # pitch, program, is_drum, start, end, track_name
                    note_sequence.append(
                        [pitch, instrument_to_program_dict[music_event[i]], is_drum,
                         current_time, current_time + duration, music_event[i]]
                    )

    return note_sequence, chord_absolute_time


if __name__ == '__main__':

    data_state = 'recover'
    # data_state = 'recover'

    with open(
            "generation_outputs/diffuseq_lmd_matched_h128_lr0.0001_t2000_sqrt_lossaware_seed102_MusicDiffuseq-ACMMM-lmd_matched-240320240407/ema_0.9999_040000.pt.samples/seed110_solverstep10_none.json",
            # "processed_data/pop909_melody_and_piano/test.jsonl",
            'r',
            encoding='utf-8'
    ) as fw:
        # injson = json.load(fw)
        for line in fw.readlines():
            dic = json.loads(line)
            data_sequence.append(dic)

    midi_name = 'generation_seq_to_midi/ACM_MusicDiffuseq/0407/' + data_state + '/'
    midi_name_number = 0

    midi_out = MidiFile(ticks_per_beat=480)
    chord_time = 0
    note_sequence = []
    for seq_dict in data_sequence:

        source = seq_dict["source"].split(" ")

        if source[7] == 'fragment_1':
            note_sequence.clear()
            midi_out.instruments.clear()
            midi_out.tempo_changes.clear()
            tempo_change = TempoChange(tempo_to_bpm_dict[source[1]], int(0))
            midi_out.tempo_changes.append(tempo_change)
            new_sequence, chord_time = tokens_to_note_event_seq(seq_dict[data_state])
            note_sequence += new_sequence
        elif source[7] == 'fragment_2' or source[7] == 'fragment_3':
            new_sequence, chord_time = tokens_to_note_event_seq(seq_dict[data_state], chord_time)
            note_sequence += new_sequence

        elif source[7] == 'fragment_4':
            new_sequence, chord_time = tokens_to_note_event_seq(seq_dict[data_state], chord_time)
            note_sequence += new_sequence

            recover_track = note_seq_to_track(note_sequence, ticks_per_beat=480)

            for track_event in recover_track:
                midi_out.instruments.append(track_event)

            midi_name_full = midi_name + str(midi_name_number) + '.mid'
            midi_name_number += 1
            print('process: ', midi_name_full)
            midi_out.dump(midi_name_full)
            print('write midi done, filename = ', midi_name_full)
