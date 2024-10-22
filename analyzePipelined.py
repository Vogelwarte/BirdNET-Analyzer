import tensorflow as tf
import numpy as np

import librosa as lr
import config as cfg
import os
import utils
import json
from pathlib import Path
from collections import defaultdict

def loadCodes():
    """Loads the eBird codes.

    Returns:
        A dictionary containing the eBird codes.
    """
    with open(os.path.join(SCRIPT_DIR, cfg.CODES_FILE), "r") as cfile:
        codes = json.load(cfile)

    return codes

SCRIPT_DIR = os.path.abspath('')
cfg.LABELS_FILE = os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)
cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
cfg.MIN_CONFIDENCE = 0.1
cfg.SIGMOID_SENSITIVITY = max(0.5, min(1.0 - (1 - 1.0), 1.5))
cfg.LABELS_FILE = os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)
cfg.LABELS = np.array(utils.readLines(cfg.LABELS_FILE))
cfg.CODES = loadCodes()

output_folder = "./example/vowa/output"
def result_writer(filename, segment_start, segment_end, predictions):
    output_file = Path(output_folder) / Path(f"{Path(filename).stem}.fast.txt")
    
    # Ensure the directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def map_fn(segment_start, segment_end, species, confidence):
        # Open the file in append mode
        with open(output_file, "a") as file:
            file.write(f"{segment_start}\t{segment_end}\t{species}\t{confidence}\n")
    [map_fn(segment_start, segment_end,*prediction) for prediction in predictions]

classifier = tf.keras.models.load_model(os.path.join(SCRIPT_DIR, cfg.MODEL_PATH), compile=True)

@tf.py_function(Tout=(tf.float32, tf.string))
def decode_flac(filename):
    audio, sr = lr.load(filename.numpy(), sr=None)
    assert sr==48000
    return (tf.convert_to_tensor(audio), filename)

@tf.function
def create_slices(audio, filename):
    snippet_length = 48000*3
    padding = tf.zeros((snippet_length - tf.shape(audio)[0] % snippet_length),dtype=tf.float32)
    padded = tf.concat([audio,padding],0)
    chunks = tf.reshape(padded,(-1, 144000))
    return tf.data.Dataset.from_tensor_slices((chunks, (tf.repeat(filename, tf.shape(chunks)[0]), tf.range(0, tf.shape(chunks)[0]))))


@tf.function
def predict(chunk, chunk_info):
    filename, window = chunk_info
    logits = classifier.basic(chunk)["scores"]
    sigmoid = 1 / (1 + tf.exp(-1 * tf.clip_by_value(logits, -15, 15)))
    return (filename, window, sigmoid)


ds = tf.data.Dataset.list_files("example/vowa/*.flac")
ds = ds.map(decode_flac,num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.interleave(create_slices, num_parallel_calls=tf.data.AUTOTUNE)

it = ds.batch(50).prefetch(tf.data.AUTOTUNE)
filenames = []
windows = []
predictions = []
for batch in it:
    b_filename, b_windows, b_sigmoids = predict(*batch)
    filenames.append(b_filename), windows.append(b_windows), predictions.append(b_sigmoids)

filenames = np.concatenate(filenames)
windows = np.concatenate(windows)
predictions = np.concatenate(predictions)

# Write the results file
# 1. Get all unique filenames of the processed data
unique_filenames = np.unique(filenames)

# create the filtered set of windows and predictions
pred_above_min_conf_flag = predictions > cfg.MIN_CONFIDENCE
arg_filtered_predictions = np.where(pred_above_min_conf_flag.any(axis=1))[0]
filtered_predictions = predictions[arg_filtered_predictions]
filtered_windows = windows[arg_filtered_predictions]
filtered_filenames = filenames[arg_filtered_predictions]
filtered_pred_above_min_conf_flag = pred_above_min_conf_flag[arg_filtered_predictions]

# 2. loop over the files
for filename in unique_filenames:
    with open(Path(output_folder) / Path(f"{Path(str(filename)).stem}.fast.txt"), "a") as output_file:
        # Write header
        output_file.write("Selection\tViewtChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tCommon Name\tSpecies Code\tConfidence\tBegin Path\tFile Offset (s)")

        # select the relevant data for the current file
        arg_current_filename = np.where(filtered_filenames == filename)
        current_file_windows = filtered_windows[arg_current_filename]
        current_file_predictions = filtered_predictions[arg_current_filename]
        current_file_pred_above_min_conf_flag = filtered_pred_above_min_conf_flag[arg_current_filename]

        # sort by window index
        arg_sorted_current_windows = np.argsort(current_file_windows)
        
        for arg_window in arg_sorted_current_windows:
            begin_time = current_file_windows[arg_window] * 3
            end_time = begin_time + 3
            
            arg_current_file_predictions = np.where(current_file_pred_above_min_conf_flag[arg_window])

            predicitons_in_window = zip(current_file_predictions[arg_window][arg_current_file_predictions],cfg.LABELS[arg_current_file_predictions])
            for confidence, label in predicitons_in_window:
                code = cfg.CODES[label] if label in cfg.CODES else label
                output_file.write(
                    f"1\tSpectrogram 1\t1\t{begin_time}\t{end_time}\t0\t24000\t{label.split('_', 1)[-1]}\t{code}\t{confidence:.4f}\t{filename}\t{begin_time}\n"
                )
                