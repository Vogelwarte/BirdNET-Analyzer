import tensorflow as tf
import numpy as np

import librosa as lr
import config as cfg
import os, glob
import utils
import json
from pathlib import Path
import math
import gc


def loadCodes():
    """Loads the eBird codes.

    Returns:
        A dictionary containing the eBird codes.
    """
    with open(os.path.join(SCRIPT_DIR, cfg.CODES_FILE), "r") as cfile:
        codes = json.load(cfile)

    return codes


SCRIPT_DIR = os.path.abspath("")
cfg.LABELS_FILE = os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)
cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
cfg.MIN_CONFIDENCE = 0.3
cfg.SIGMOID_SENSITIVITY = max(0.5, min(1.0 - (1 - 1.0), 1.5))
cfg.LABELS_FILE = os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)
cfg.LABELS = np.array(utils.readLines(cfg.LABELS_FILE))
cfg.CODES = loadCodes()

root_folder = Path("/srv/ext/recordings/420650_AFP_Weissrückensprecht/")
output_folder = Path("/srv/results/420650_AFP_Weissrückensprecht")
classifier = tf.keras.models.load_model( 
    os.path.join(SCRIPT_DIR, cfg.MODEL_PATH), compile=True
)

cfg.SPECIES_FILE = utils.readLines(output_folder / Path("species_list.txt"))
#cfg.SPECIES_FILE = None


@tf.py_function(Tout=(tf.float32, tf.string))
def decode_flac(filename):
    audio, sr = lr.load(filename.numpy(), sr=None)
    if sr != 48000:
        lr.resample(audio,orig_sr=sr,target_sr=48000)
    filename = os.path.relpath(filename.numpy().decode("utf-8"),root_folder) # just return the path relative to the root folder
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
    return [array[i*batch_size:i*batch_size+batch_size] for i in range(len(array)//batch_size+1)]

file_list = np.array([file for file in glob.glob(os.path.join(root_folder, '**', f'*.flac'), recursive=True)])
for ds in split_in_batches(file_list,100):
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.map(decode_flac, num_parallel_calls=5)
    ds = ds.interleave(create_slices, block_length=500, cycle_length=5)

    it = ds.batch(500).prefetch(5)
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
        output_file_path = Path(output_folder) / filename.parent / Path(f"{filename.stem}.GPU.BirdNET.selection.table.txt")
        os.makedirs(output_file_path.parent,exist_ok=True) # create directory if it does not already exist
        with open(output_file_path, "w") as output_file:
            index = 1
            # Write header
            output_file.write(
                "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tCommon Name\tSpecies Code\tConfidence\tBegin Path\tFile Offset (s)\n"
            )

            # select the relevant data for the current file
            arg_current_filename = np.where(filtered_filenames == bytes(filename)) #the filtered_filenames are still bytes
            current_file_windows = filtered_windows[arg_current_filename]
            current_file_predictions = filtered_predictions[arg_current_filename]
            current_file_pred_above_min_conf_flag = filtered_pred_above_min_conf_flag[
                arg_current_filename
            ]

            # sort by window index
            arg_sorted_current_windows = np.argsort(current_file_windows)

            for arg_window in arg_sorted_current_windows:
                begin_time = current_file_windows[arg_window] * 3
                end_time = begin_time + 3

                arg_current_file_predictions = current_file_pred_above_min_conf_flag[
                    arg_window
                ]

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

