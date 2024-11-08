import tensorflow as tf
import numpy as np
import librosa as lr
import audioread  # alternative to default soundfile from librosa
import resampy
import os, glob
import json
from pathlib import Path
import math
import argparse
import gc

import birdnet_analyzer.config as cfg
import birdnet_analyzer.utils as utils
import soundfile

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def loadCodes():
    """Loads the eBird codes.

    Returns:
        A dictionary containing the eBird codes.
    """
    with open(os.path.join(SCRIPT_DIR, cfg.CODES_FILE), "r") as cfile:
        codes = json.load(cfile)

    return codes


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Analyze audio files with BirdNET on GPU"
    )
    parser.add_argument("--i", help="Path to input file or folder.")
    parser.add_argument("--o", help="Path to output folder.")
    parser.add_argument(
        "--slist",
        default="",
        help='Path to species list file or folder. If folder is provided, species list needs to be named "species_list.txt". If lat and lon are provided, this list will be ignored.',
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.1,
        help="Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="This is currently being ignored!"
    )
    args = parser.parse_args()

    cfg.LABELS_FILE = os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)
    cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
    # Set confidence threshold
    cfg.MIN_CONFIDENCE = max(0.01, min(0.99, float(args.min_conf)))
    cfg.SIGMOID_SENSITIVITY = max(0.5, min(1.0 - (1 - 1.0), 1.5))
    cfg.LABELS_FILE = os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)
    cfg.LABELS = np.array(utils.readLines(cfg.LABELS_FILE))
    cfg.CODES = loadCodes()

    root_folder = Path(args.i)
    output_folder = Path(args.o)
    classifier = tf.keras.models.load_model(
        os.path.join(SCRIPT_DIR, cfg.PB_MODEL), compile=True
    )

    cfg.SPECIES_FILE = utils.readLines(args.slist)

    def lib_audioread_load(filename, sr=48000):
        result_audioread = []
        with audioread.audio_open(filename) as f:
            for i, buf in enumerate(f):
                result_audioread.append(np.frombuffer(buf, dtype=np.short))
        result_audioread = np.concatenate(result_audioread)
        result_audioread = lr.util.buf_to_float(result_audioread)
        resampy.resample(result_audioread, f.samplerate, sr)
        return result_audioread

    @tf.py_function(Tout=(tf.float32, tf.string))
    def decode_flac(filename):
        print(f"Analyzing: {filename.numpy().decode('utf-8')}")
        audio, sr = np.array([0.0], dtype=np.float32), 48000
        try:
            audio, sr = lr.load(filename.numpy(), sr=48000)
        except Exception as _:
            print(
                f"INFO: Could not open {filename.numpy().decode('utf-8')} with soundfile, trying audioread instead"
            )
            try:
                audio = lib_audioread_load(filename.numpy().decode("utf-8"))
            except Exception as ex:
                print(f"WARNING: Ignoring {filename} due to error {ex}")
                utils.writeErrorLog(ex)
        filename = os.path.relpath(
            filename.numpy().decode("utf-8"), root_folder
        )  # just return the path relative to the root folder
        return (tf.convert_to_tensor(audio), filename)

    @tf.function
    def create_slices(audio, filename):
        snippet_length = 48000 * 3
        padding = tf.zeros(
            (snippet_length - tf.shape(audio)[0] % snippet_length), dtype=tf.float32
        )
        padded = tf.concat([audio, padding], 0)
        chunks = tf.reshape(padded, (-1, 144000))
        return tf.data.Dataset.from_tensor_slices(
            (
                chunks,
                (
                    tf.repeat(filename, tf.shape(chunks)[0]),
                    tf.range(0, tf.shape(chunks)[0]),
                ),
            )
        )

    @tf.function
    def predict(chunk, chunk_info):
        filename, window = chunk_info
        logits = classifier.basic(chunk)["scores"]
        return (filename, window, logits)

    def split_in_batches(array, batch_size):
        return [
            array[i * batch_size : i * batch_size + batch_size]
            for i in range(len(array) // batch_size + 1)
        ]

    file_list = np.array(
        [
            file
            for file in glob.glob(
                os.path.join(root_folder, "**", f"*.flac"), recursive=True
            )
        ]
    )
    for ds in split_in_batches(file_list, 400):
        ds = tf.data.Dataset.from_tensor_slices(ds)
        ds = ds.map(decode_flac, num_parallel_calls=10, deterministic=False).filter(
            lambda audio, filename: len(audio) > 1
        )
        ds = ds.interleave(create_slices, block_length=500, cycle_length=5)

        it = ds.prefetch(500).batch(500).prefetch(5)
        filenames = []
        windows = []
        predictions = []
        for batch in it:
            b_filename, b_windows, b_sigmoids = predict(*batch)
            (
                filenames.append(b_filename),
                windows.append(b_windows),
                predictions.append(b_sigmoids),
            )

        del ds
        del it
        gc.collect()

        filenames = np.concatenate(filenames)
        windows = np.concatenate(windows)
        predictions = np.concatenate(predictions)

        # Write the results file
        # 1. Get all unique filenames of the processed data
        unique_filenames = np.unique(filenames)

        # create the filtered set of windows and predictions
        pred_above_min_conf_flag = predictions > -math.log(
            (1 - cfg.MIN_CONFIDENCE) / cfg.MIN_CONFIDENCE
        )
        arg_filtered_predictions = np.where(pred_above_min_conf_flag.any(axis=1))[0]
        filtered_predictions = predictions[arg_filtered_predictions]
        filtered_windows = windows[arg_filtered_predictions]
        filtered_filenames = filenames[arg_filtered_predictions]
        filtered_pred_above_min_conf_flag = pred_above_min_conf_flag[
            arg_filtered_predictions
        ]

        # 2. loop over the files
        for filename in unique_filenames:
            filename = Path(filename.decode("utf-8"))
            output_file_path = (
                Path(output_folder)
                / filename.parent
                / Path(f"{filename.stem}.GPU.BirdNET.selection.table.txt")
            )
            os.makedirs(
                output_file_path.parent, exist_ok=True
            )  # create directory if it does not already exist
            with open(output_file_path, "w") as output_file:
                index = 1
                # Write header
                output_file.write(
                    "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tCommon Name\tSpecies Code\tConfidence\tBegin Path\tFile Offset (s)\n"
                )

                # select the relevant data for the current file
                arg_current_filename = np.where(
                    filtered_filenames == bytes(filename)
                )  # the filtered_filenames are still bytes
                current_file_windows = filtered_windows[arg_current_filename]
                current_file_predictions = filtered_predictions[arg_current_filename]
                current_file_pred_above_min_conf_flag = (
                    filtered_pred_above_min_conf_flag[arg_current_filename]
                )

                # sort by window index
                arg_sorted_current_windows = np.argsort(current_file_windows)

                for arg_window in arg_sorted_current_windows:
                    begin_time = current_file_windows[arg_window] * 3
                    end_time = begin_time + 3

                    arg_current_file_predictions = (
                        current_file_pred_above_min_conf_flag[arg_window]
                    )

                    predicitons_in_window = sorted(
                        zip(
                            1
                            / (
                                1
                                + np.exp(
                                    -current_file_predictions[arg_window][
                                        arg_current_file_predictions
                                    ]
                                )
                            ),
                            cfg.LABELS[arg_current_file_predictions],
                        ),
                        key=lambda x: x[0],
                        reverse=True,
                    )
                    for confidence, label in predicitons_in_window:
                        if not cfg.SPECIES_FILE or label in cfg.SPECIES_FILE:
                            code = cfg.CODES[label] if label in cfg.CODES else label
                            output_file.write(
                                f"{index}\tSpectrogram 1\t1\t{begin_time:.1f}\t{end_time:.1f}\t0\t15000\t{label.split('_', 1)[-1]}\t{code}\t{confidence:.4f}\t{filename}\t{begin_time:.1f}\n"
                            )
                            index = index + 1
