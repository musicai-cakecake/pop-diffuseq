import miditoolkit
from .chords import ChordsDetector


def detect_chords_test():
    midi_path = "/data/jingcheng.wu/test/leading_sheet/1084.mid"
    mf = miditoolkit.midi.MidiFile(midi_path)
    all_notes = mf.instruments[1].notes
    all_notes.sort(key=lambda x: (x.start, x.pitch))
    temp_chords = ChordsDetector(all_notes, mf.ticks_per_beat).detect_chords_by_chorder()  # pretty_midi的默认resolution为220
    print("done")