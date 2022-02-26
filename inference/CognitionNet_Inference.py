import pickle
import sys

sys.path.append("../")

import numpy as np
import pandas as pd
import tensorflow as tf
from config import *
from models.CognitionNet import *
from sklearn.metrics import confusion_matrix
from training.Train_CognitionNet import *
from utils import *

tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)
tf.get_logger().setLevel("ERROR")


class CognitionNet_Inference:
    def __init__(self, folder, model_loc1, model_loc2, model_loc3, model_loc4):
        """
        Load model weights along with cluster centers for kmeans clustering
        """
        self.classifier = tf.keras.models.load_model(model_loc1, compile=True)
        self.game_enc = tf.keras.models.load_model(model_loc2, compile=True)
        self.transition_model_loaded = tf.keras.models.load_model(model_loc3)
        with open(model_loc4, "rb") as f:
            self.kmeans_model = pickle.load(f)
        self.inference_data_path = folder

    def get_transition_matrix(self, x):
        input_ = []
        for sequence in x:
            matrix = x = np.zeros((K1, K1))
            for i in range(len(sequence) - 1):
                matrix[int(sequence[i])][int(sequence[i + 1])] += 1
            input_.append(matrix / np.sum(matrix))
        input_ = np.asarray(input_)
        return input_

    def get_cluster_allocation_v2(self, input_data):
        """
        Predict clusters for each game sequence
        """
        latent_space, encoder_states_1, encoder_states_2, encoder_states_3 = self.game_enc(
            input_data, False
        )
        input_kmeans = tf.concat(
            [tf.concat([encoder_states_1[0], encoder_states_2[0]], 1), encoder_states_3[0]], 1
        ).numpy()
        input_kmeans = input_kmeans.astype("float")
        predicted_clusters = self.kmeans_model.predict(input_kmeans)
        return latent_space, predicted_clusters

    def predict_user_journey(self, inference_chunks):
        """
        Prepare data for inference
        """
        for ichunk in inference_chunks:
            x_inf, class_inf, sample_inf, seq_inf = data_manipulation(
                self.inference_data_path, ichunk
            )
            class_inf = pd.Series(class_inf.reshape(class_inf.shape[0]))
            sample_inf = pd.Series(sample_inf)
            seq_inf = pd.Series(seq_inf)
            process_size = 6000
            y_1, cluster_ids = self.get_cluster_allocation_v2(x_inf[:process_size, :, :])
            index = process_size

            while index < x_inf.shape[0]:
                if index + process_size < x_inf.shape[0]:
                    y_1_subpart, cluster_ids_subpart = self.get_cluster_allocation_v2(
                        x_inf[index : index + process_size, :, :]
                    )
                else:
                    y_1_subpart, cluster_ids_subpart = self.get_cluster_allocation_v2(
                        x_inf[index : x_inf.shape[0], :, :]
                    )
                y_1 = np.append(y_1, y_1_subpart, axis=0)
                cluster_ids = np.append(cluster_ids, cluster_ids_subpart, axis=0)
                index = index + process_size

            df_ = pd.concat(
                [class_inf, sample_inf, seq_inf, pd.Series(cluster_ids)],
                axis=1,
                keys=["class_id", "sample#", "seq#", "cluster_id"],
            )
            x_2 = []
            y_2_actual = []
            df_["seq#"] = pd.to_numeric(df_["seq#"])
            for class_ in df_["class_id"].unique():
                for sample_ in df_[df_["class_id"] == class_]["sample#"].unique():
                    cluster_list = (
                        df_[(df_["class_id"] == class_) & (df_["sample#"] == sample_)]
                        .sort_values(by="seq#", ascending=True)["cluster_id"]
                        .values
                    )
                    x_2.append(cluster_list)
                    y_2_actual.append([class_])
            y_2_actual = np.asarray(y_2_actual)
            x_transition = self.get_transition_matrix(x_2)
            prediction, _ = self.transition_model_loaded(
                x_transition.reshape(
                    x_transition.shape[0], x_transition.shape[1], x_transition.shape[2], 1
                ),
                False,
            )
            y_pred_inf = np.argmax(prediction.numpy(), axis=1)
            mapping = {"sustainers": 0, "burnouts": 1, "churnouts": 2}
            y_true_inf = [mapping[item[0]] for item in y_2_actual]
            print(confusion_matrix(y_true_inf, y_pred_inf))
            return df_, y_pred_inf


if __name__ == "__main__":

    # Load saved model weights location
    model_loc1 = "../modelweights/CognitionNet_TM/Classifier_model"
    model_loc2 = "../modelweights/CognitionNet_TM/Encoder_model"
    model_loc3 = "../modelweights/CognitionNet_TM/Transition_model"
    model_loc4 = "../modelweights/CognitionNet_TM/Kmeans_model/kmean_clustering.pkl"

    infer_cognitionNet = CognitionNet_Inference(
        data_npy_folder_test, model_loc1, model_loc2, model_loc3, model_loc4
    )
    infer_cognitionNet.predict_user_journey([2])
