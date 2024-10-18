import tensorflow as tf
import numpy as np

import librosa as lr
import config as cfg
import os
import utils
import operator
from pathlib import Path

import numba as nb

tf.config.list_physical_devices()
SCRIPT_DIR = os.path.abspath('')
cfg.LABELS_FILE = os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)
cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
cfg.MIN_CONFIDENCE = 0.5
cfg.SIGMOID_SENSITIVITY = max(0.5, min(1.0 - (1 - 1.0), 1.5))
cfg.LABELS_FILE = os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)
cfg.LABELS = utils.readLines(cfg.LABELS_FILE)


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

def assemble_result(filenames, windows, batch_predictions):
    def map_fn(filename, window, predictions):
        s_start = window * 3
        s_end = s_start + 3
        p_labels = filter(lambda x: x[1] > cfg.MIN_CONFIDENCE, zip(cfg.LABELS, predictions))
        return sorted(p_labels, key=operator.itemgetter(1), reverse=True)
        #result_writer(str(filename), s_start, s_end, p_sorted)
    
    list(map(lambda x: map_fn(*x),iter(zip(filenames, windows, batch_predictions))))
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
ds = ds.interleave(create_slices, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

batched_results = []
for filenames, windows, predictions in ds.batch(50).map(predict).as_numpy_iterator():
    batched_results.append(assemble_result(filenames, windows, predictions))

batched_results