import sys

sys.path.append("../")

import tensorflow as tf
from config import *
from models.CognitionNet import *
from utils import *

from training.Train_CognitionNet import *

tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)
tf.get_logger().setLevel("ERROR")


if __name__ == "__main__":
    print("Starting training of CognitionNet")
    trainCognitionNet(no_collab_epochs=2, numchunks=1)
    print("Completed training of CognitionNet")

