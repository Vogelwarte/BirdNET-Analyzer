import tensorflow as tf
import numpy as np

import librosa as lr
import config as cfg
import os
import utils
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = os.path.abspath('')
cfg.LABELS_FILE = os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)
cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
cfg.MIN_CONFIDENCE = 0.5
cfg.SIGMOID_SENSITIVITY = max(0.5, min(1.0 - (1 - 1.0), 1.5))
cfg.LABELS_FILE = os.path.join(SCRIPT_DIR, cfg.LABELS_FILE)
cfg.LABELS = np.array(utils.readLines(cfg.LABELS_FILE))



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

#def gather_result(filenames: np.ndarray, windows: np.ndarray, batch_predictions:np.ndarray):
    #relevant_predictions = batch_predictions > cfg.MIN_CONFIDENCE

    

    # for filename, window, predicitons in zip(filenames, windows, batch_predictions):
    #     s_start = window * 3
    #     s_end = s_start + 3
    #     p_labels = filter(lambda x: x[1] > cfg.MIN_CONFIDENCE, zip(cfg.LABELS, predictions))
    #     p_sorted = sorted(p_labels, key=operator.itemgetter(1), reverse=True)
    #     gather.append((filename, s_start, s_end, p_sorted))
    # return gather

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

# @tf.function
# def (filename, window, sigmoid):
    


ds = tf.data.Dataset.list_files("example/vowa/*.flac")
ds = ds.map(decode_flac,num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.interleave(create_slices, num_parallel_calls=tf.data.AUTOTUNE)

# res = []
# for batch in ds.batch(50).prefetch(tf.data.AUTOTUNE):
#     res.append(predict(*batch))

# # for batch in ds.batch(50).map(predict, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE):
# #     res.append(batch)


# print(res)

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

print(filenames.shape)
print(windows.shape)
print(predictions.shape)
# for b_filename, b_windows, b_sigmoids in ds.batch(50).map(predict).prefetch(tf.data.AUTOTUNE):
#     filenames.append(b_filename), windows.append(b_windows), sigmoids.append(b_sigmoids)


# result_dict = defaultdict(list)
 
# # List comprehension to perform the same operation
# x = [ result_dict[filename.numpy()].append((window, predictions)) for filename, window, predictions in it ]

# for filename, window, predictions in it:
#     result_dict[filename].append((window, predictions))


# #[ result_dict[filename].append((window,predictions)) if (filename in result_dict) else result_dict = {**result_dict, filename: [(window, predictions)]} for filename, window, predictions in it ]
# result_dict = {}
# def gather_result(filename, window, predictions, result_dict):
#     if filename in result_dict:
#         result_dict[filename].append((window, predictions))
#     else:
#         result_dict[filename] = [(window, predictions)]

# list(map(lambda x: gather_result(*x, result_dict), it))

