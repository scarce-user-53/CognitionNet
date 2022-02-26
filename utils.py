import sys

import numpy as np
import tensorflow as tf

sys.path.append("../")

from config import *

tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)
tf.get_logger().setLevel("ERROR")


@tf.function
def trace_loss(H):
    FtHt = tf.matmul(tf.transpose(style_model.F), tf.transpose(H))
    HF = tf.matmul(H, style_model.F)
    trace_loss = tf.linalg.trace(tf.matmul(np.transpose(H), H)) - tf.linalg.trace(
        tf.matmul(FtHt, HF)
    )
    return trace_loss


@tf.function
def categorical_loss(y_true, y_pred):
    mapping = {"sustainers": 0, "burnouts": 1, "churnouts": 2}
    y_true = np.array([mapping[item[0]] for item in y_true])
    y_true = y_true.reshape(y_true.shape[0], 1)
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    return scce(y_true, y_pred)


def data_manipulation(data_folder, data_file_no):
    """
    Reading the npy file and obtaining the sequences per sample per class.
    The first network only deals with sequences.
    The second network is supervised and needs to know the sequence ordering.
    """
    data_file = data_folder + "chunk_" + str(data_file_no) + ".npy"
    data_chunk = np.load(data_file, allow_pickle=True)
    x = data_chunk[:, :window_length, :-7]
    # Selecting some features
    x = x[:, :, primary_list]
    class_ = data_chunk[:, :window_length, -3][:, 0]
    sample_ = data_chunk[:, :window_length, -2][:, 0]
    seq_ = data_chunk[:, :window_length, -1][:, 0]
    return (
        np.asarray(x).astype("float64"),
        np.asarray(class_.reshape(class_.shape[0], 1)),
        np.asarray(sample_).astype("float64"),
        np.asarray(seq_).astype("float64"),
    )

