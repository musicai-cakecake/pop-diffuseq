import os
import json
import argparse
import numpy as np
from core import PITCH_CLASSES
from core import get_num_notes_one_instruments, get_used_pitch_one_instrument, get_pitch_class_histogram
from core import get_avg_duration_beats, probability_of_pitch_transition_destination, get_avg_ioi_beats
from core import get_pitch_class_transition_matrix
from core import get_avg_ioi
import matplotlib.pyplot as plt
import pretty_midi


def evaluate_single_midi(midi_filepath, return_numpy=False):

    pretty_midi_features = pretty_midi.PrettyMIDI(midi_filepath)

    # num_notes_piano_roll = get_num_notes_one_instruments(pretty_midi_features)

    # 获得一个轨道的所有使用过的音符
    # used_pitch = get_used_pitch_one_instrument(pretty_midi_features)

    pitch_class_histogram = get_pitch_class_histogram(pretty_midi_features)
    # Histogram of 12 pitch, with weighted duration shape 12

    pitch_class_transition_matrix = get_pitch_class_transition_matrix(pretty_midi_features, normalize=3)
    # transition matrix of 12 pitch，从行到列的转移矩阵
    # normalize=2 对整体做 L2 normalization，主要观察列，表示其他音高转移到某个音高的概率
    # normalize=3 对每个行做 L2 normalization，主要观察行，表示某个音高 转移到其他音高的概率

    # p_pitch_destination = probability_of_pitch_transition_destination(pretty_midi_features)
    # p_pitch_destination_sort = np.sort(p_pitch_destination)  # 从小到大排序，位置-1是最大的值
    # p_pitch_destination_idx_sort = np.argsort(p_pitch_destination)  # 数值从小到大的索引，位置-1是最大值的索引

    # To calculate the inter-onset-interval in the symbolic music domain.
    # avg_inter_onset_interval = get_avg_ioi(pretty_midi_features)
    avg_inter_onset_interval_beats = get_avg_ioi_beats(midi_filepath)

    avg_duration_in_beat = get_avg_duration_beats(midi_filepath)

    metrics_one_midi = {
        # 'num_notes': num_notes_piano_roll,
        # 'used_pitch': used_pitch,
        'pitch_class_histogram': pitch_class_histogram,
        'pitch_class_transition_matrix': pitch_class_transition_matrix,
        'avg_ioi': avg_inter_onset_interval_beats,
        'avg_dur': avg_duration_in_beat
    }

    if return_numpy:
        return metrics_one_midi
    for key in metrics_one_midi.keys():
        if isinstance(metrics_one_midi[key], (np.ndarray, np.generic)):
            metrics_one_midi[key] = metrics_one_midi[key].tolist()
    return metrics_one_midi


def evaluate_single_track(midi_filepath, track_name=None, return_numpy=False):

    pretty_midi_features = pretty_midi.PrettyMIDI(midi_filepath)

    # num_notes_piano_roll = get_num_notes_one_instruments(pretty_midi_features)

    # 获得一个轨道的所有使用过的音符
    # used_pitch = get_used_pitch_one_instrument(pretty_midi_features)

    pitch_class_histogram = get_pitch_class_histogram(pretty_midi_features)
    # Histogram of 12 pitch, with weighted duration shape 12

    pitch_class_transition_matrix = get_pitch_class_transition_matrix(pretty_midi_features, normalize=2)
    # transition matrix of 12 pitch，从行到列的转移矩阵
    # normalize=2 对整体做 L2 normalization，主要观察列，表示其他音高转移到某个音高的概率
    # normalize=3 对每个行做 L2 normalization，主要观察行，表示某个音高 转移到其他音高的概率

    # p_pitch_destination = probability_of_pitch_transition_destination(pretty_midi_features)
    # p_pitch_destination_sort = np.sort(p_pitch_destination)  # 从小到大排序，位置-1是最大的值
    # p_pitch_destination_idx_sort = np.argsort(p_pitch_destination)  # 数值从小到大的索引，位置-1是最大值的索引

    # To calculate the inter-onset-interval in the symbolic music domain.
    # avg_inter_onset_interval = get_avg_ioi(pretty_midi_features)
    avg_inter_onset_interval_beats = get_avg_ioi_beats(midi_filepath)

    avg_duration_in_beat = get_avg_duration_beats(midi_filepath)

    metrics_one_midi = {
        # 'num_notes': num_notes_piano_roll,
        # 'used_pitch': used_pitch,
        'pitch_class_histogram': pitch_class_histogram,
        'pitch_class_transition_matrix': pitch_class_transition_matrix,
        'avg_ioi': avg_inter_onset_interval_beats,
        'avg_dur': avg_duration_in_beat
    }

    if return_numpy:
        return metrics_one_midi
    for key in metrics_one_midi.keys():
        if isinstance(metrics_one_midi[key], (np.ndarray, np.generic)):
            metrics_one_midi[key] = metrics_one_midi[key].tolist()
    return metrics_one_midi


def plot_pitch_class_histogram(pitch_class_histogram, save_path):
    fig, ax = plt.subplots(1)
    ax.bar(PITCH_CLASSES, height=pitch_class_histogram)
    fig.savefig(save_path)
    plt.close(fig)


def plot_pitch_class_transition_matrix(pitch_class_transition_matrix, save_path):
    fig, ax = plt.subplots(1)
    ax.set_xticks(np.arange(len(PITCH_CLASSES)), labels=PITCH_CLASSES)
    ax.set_yticks(np.arange(len(PITCH_CLASSES)), labels=PITCH_CLASSES)
    ax.imshow(pitch_class_transition_matrix, norm=None)
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in single midi evaluation.')
    parser.add_argument(
        '-midi-path', type=str,
        help='The midi file to evaluate'
    )
    parser.add_argument(
        '-out-dir', type=str, default="./results",
        help='The output directory to save metrics'
    )
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    _, midi_name = os.path.split(args.midi_path)
    midi_name = os.path.splitext(midi_name)[0]

    metrics = evaluate_single_midi(args.midi_path, return_numpy=False)

    out_json_filename = midi_name + '_metrics.json'
    out_pctm_filename = midi_name + '_pctm.pdf'
    out_pitch_hist_filename = midi_name + '_pitch_hist.pdf'

    out_json_filepath = os.path.join(args.out_dir, out_json_filename)
    out_pctm_filepath = os.path.join(args.out_dir, out_pctm_filename)
    out_pitch_hist_filepath = os.path.join(args.out_dir, out_pitch_hist_filename)

    with open(out_json_filepath, "w") as outfile:
        json.dump(metrics, outfile)

    plot_pitch_class_transition_matrix(
        metrics["pitch_class_transition_matrix"],
        out_pctm_filepath
    )
    plot_pitch_class_histogram(
        metrics["pitch_class_histogram"],
        out_pitch_hist_filepath
    )
    print("Saved metrics to {}".format(args.out_dir))
