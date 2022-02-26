import os
import sys

sys.path.append("../")
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from config import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow import keras
from tensorflow.keras.layers import (
    LSTM,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from utils import *

tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)
tf.get_logger().setLevel("ERROR")

# Check for available GPU devices
print(tf.config.experimental.list_physical_devices("GPU"))


class Transition_model(keras.Model):
    """
    The interpretor network to help to classify a user wrt to transition matrix
    """

    def __init__(self, k, layer_dim):
        super(Transition_model, self).__init__()
        self.K1 = k
        self.class_outputs = classes
        self.feature_space = layer_dim
        self.conv2d_layer_1 = Conv2D(
            16, (2, 2), activation="relu", input_shape=(self.K1, self.K1, 1)
        )
        self.conv2d_layer_2 = Conv2D(8, (3, 3), activation="relu")
        self.max_pool = MaxPooling2D(pool_size=(2, 2))
        self.dropout = Dropout(0.25)
        self.flatten = Flatten()
        self.Dense_layer_1 = Dense(self.feature_space, activation="relu")
        self.Dense_layer_2 = Dense(self.class_outputs, activation="softmax")

    def call(self, input_data):
        conv2d_layer_1_output = self.conv2d_layer_1(input_data)
        conv2d_layer_2_output = self.conv2d_layer_2(conv2d_layer_1_output)
        max_pool_output = self.max_pool(conv2d_layer_2_output)
        dropout_output = self.dropout(max_pool_output)
        flatten_output = self.flatten(dropout_output)
        Dense_layer_1_output = self.Dense_layer_1(flatten_output)
        Dense_layer_2_output = self.Dense_layer_2(Dense_layer_1_output)
        return Dense_layer_2_output, Dense_layer_1_output

    def get_transition_matrix(self, x):
        """
        Prepare trasition matrix from sequence of clusters transition per user
        """
        input_ = []
        sign_matrix = []
        for sequence in x:
            assert nseq_per_user == len(sequence)
            user_matrix = np.ones((len(sequence), len(sequence)))
            matrix = np.zeros((self.K1, self.K1))
            for i in range(len(sequence) - 1):
                matrix[int(sequence[i])][int(sequence[i + 1])] += 1
                for j in range(i, len(sequence)):
                    if sequence[i] == sequence[j]:
                        user_matrix[int(sequence[i])][int(sequence[j])] = -1
                        user_matrix[int(sequence[j])][int(sequence[i])] = -1

            input_.append(matrix / np.sum(matrix))
            sign_matrix.append(user_matrix)
        input_ = np.asarray(input_)
        sign_matrix = np.asarray(sign_matrix)
        return input_, sign_matrix

    def get_metadata_for_transition(self, class_users, sample_users, seq_users, cluster_ids):
        """
        Get cluster allocations for each continuous play of users
        """
        class_users = pd.Series(class_users.reshape(class_users.shape[0]))
        sample_users = pd.Series(sample_users)
        seq_users = pd.Series(seq_users)
        df_ = pd.concat(
            [class_users, sample_users, seq_users, pd.Series(cluster_ids)],
            axis=1,
            keys=["class_id", "sample#", "seq#", "cluster_id"],
        )
        x_2 = []
        y_2_actual = []
        users = []
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
                users.append(sample_)
        y_2_actual = np.asarray(y_2_actual)
        x_transition, pr_matrix = self.get_transition_matrix(x_2)
        return [x_transition, y_2_actual, np.array(users), pr_matrix]


class Game_Behavior_Encoder(keras.Model):
    """
    Encoder helps to prepare the latent space for each gameplay sequence
    """

    def __init__(self, num_class=4, window_length=50, num_features=8, k=5, latent_dim=None):
        super(Game_Behavior_Encoder, self).__init__()
        self.window_length = window_length
        self.num_features = num_features
        self.K1 = k
        if latent_dim is None:
            self.latent_dim = self.num_features - 2
        else:
            self.latent_dim = latent_dim
        self.num_class = num_class
        self.kmeans_cluster = KMeans(n_clusters=K1, random_state=0)
        self.transition_model = Transition_model(self.K1, fm_space)
        initializer = tf.keras.initializers.Orthogonal()
        self.F = tf.Variable(
            initializer(shape=(users_seq2seq_phase2 * nseq_per_user, K1)), trainable=False
        )
        self.encoder_inputs = Input(
            shape=(self.window_length, self.num_features), name="encoder_input"
        )
        self.encoder_lstm_stack_1 = LSTM(64, return_sequences=True, return_state=True)
        self.encoder_lstm_stack_2 = LSTM(32, return_sequences=True, return_state=True)
        self.encoder_lstm_stack_3 = LSTM(
            self.num_features, return_sequences=True, return_state=True
        )
        # Below layers are used for Bridge Loss
        self.W_b = Dense(1, activation="relu")
        self.W_c = Dense(fm_space, activation="relu")

        self.TraceLoss_history = []
        self.RC_history = []
        self.BridgeLoss_history = []
        self.Silhouttescore_history = []

    @tf.function
    def custom_loss_MSE(self, y_labels, y_pred):
        mse_loss = K.mean(K.square(tf.cast(y_labels, tf.float32) - y_pred))
        return mse_loss

    def call(self, input_data):
        encoder_outputs_1, state_h_1, state_c_1 = self.encoder_lstm_stack_1(input_data)
        encoder_outputs_2, state_h_2, state_c_2 = self.encoder_lstm_stack_2(encoder_outputs_1)
        encoder_outputs_3, state_h_3, state_c_3 = self.encoder_lstm_stack_3(encoder_outputs_2)
        return (
            encoder_outputs_3,
            [state_h_1, state_c_1],
            [state_h_2, state_c_2],
            [state_h_3, state_c_3],
        )

    def create_cluster(self, latent_space):
        self.kmeans_cluster.fit(latent_space)

    def predict_cluster(self, latent_space):
        return self.kmeans_cluster.predict(latent_space.numpy().astype("float"))

    def get_silhoutte_score(self, latent_space):
        s_score = silhouette_score(latent_space, self.kmeans_cluster.predict(latent_space))
        return s_score


class Game_Behavior_Decoder(keras.Model):
    """
    Decoder helps to reconstruct the gameplay features from the latent space
    """

    def __init__(self, num_class=4, window_length=50, num_features=8, k=5, latent_dim=None):
        super(Game_Behavior_Decoder, self).__init__()
        self.window_length = window_length
        self.num_features = num_features
        self.K1 = k
        if latent_dim is None:
            self.latent_dim = self.num_features - 2
        else:
            self.latent_dim = latent_dim
        self.num_class = num_class
        self.decoder_lstm_stack_1 = LSTM(64, return_sequences=True, return_state=True)
        self.decoder_lstm_stack_2 = LSTM(32, return_sequences=True, return_state=True)
        self.decoder_lstm_stack_3 = LSTM(
            self.num_features, return_sequences=True, return_state=True
        )
        # Used for attention
        self.W1 = tf.keras.layers.Dense(self.num_features)
        self.W2 = tf.keras.layers.Dense(self.num_features)
        self.V = tf.keras.layers.Dense(1)

    def call(
        self, decoder_input, decoder_states_1, decoder_states_2, decoder_states_3, enc_output
    ):

        hidden = decoder_states_3[0]
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        x = tf.concat([tf.expand_dims(context_vector, 1), decoder_input], axis=-1)

        (
            decoder_outputs_1,
            hidden_state_decoder_1,
            cell_state_decoder_1,
        ) = self.decoder_lstm_stack_1(x, initial_state=decoder_states_1)
        (
            decoder_outputs_2,
            hidden_state_decoder_2,
            cell_state_decoder_2,
        ) = self.decoder_lstm_stack_2(decoder_outputs_1, initial_state=decoder_states_2)
        decoder_outputs, hidden_state_decoder_3, cell_state_decoder_3 = self.decoder_lstm_stack_3(
            decoder_outputs_2, initial_state=decoder_states_3
        )
        decoder_outputs = tf.reshape(decoder_outputs, (-1, decoder_outputs.shape[2]))
        return (
            decoder_outputs,
            [hidden_state_decoder_1, cell_state_decoder_1],
            [hidden_state_decoder_2, cell_state_decoder_2],
            [hidden_state_decoder_3, cell_state_decoder_3],
            attention_weights,
        )


class Play_Style_Transition(tf.keras.Model):
    """
    For training of the transition network with all the cluster allocation of sequences for each user
    """

    def __init__(self, game_style_encoder, k, num_classes, F_shape):
        super(Play_Style_Transition, self).__init__()
        self.K1 = k
        self.topics = num_classes
        self.game_style_encoder = game_style_encoder
        self.initializer = tf.keras.initializers.Orthogonal()
        self.F = tf.Variable(self.initializer(shape=(F_shape, self.K1)), trainable=False)
        self.transition_model = self.game_style_encoder.transition_model
        self.Trace_loss = []
        self.Sparse_CCE = []
        self.CCE_transition_model_independent = []
        self.ls = []

    def get_cluster_allocation(self, input_data):
        (
            latent_space,
            encoder_states_1,
            encoder_states_2,
            encoder_states_3,
        ) = self.game_style_encoder(input_data)
        input_kmeans = tf.concat(
            [tf.concat([encoder_states_1[0], encoder_states_2[0]], 1), encoder_states_3[0]], 1
        ).numpy()
        input_kmeans = input_kmeans.astype("float")
        predicted_clusters = self.game_style_encoder.kmeans_cluster.predict(input_kmeans)
        return latent_space, predicted_clusters

    def get_transition_matrix(self, x):
        input_ = []
        for sequence in x:
            matrix = x = np.zeros((self.K1, self.K1))
            for i in range(len(sequence) - 1):
                matrix[int(sequence[i])][int(sequence[i + 1])] += 1
            input_.append(matrix / np.sum(matrix))
        input_ = np.asarray(input_)

        return input_

    def get_data(self, json_part, data_folder, process_size):
        x_1, class_users, sample_users, seq_users = data_manipulation(data_folder, json_part)
        class_users = pd.Series(class_users.reshape(class_users.shape[0]))
        sample_users = pd.Series(sample_users)
        seq_users = pd.Series(seq_users)
        y_1, cluster_ids = self.get_cluster_allocation(x_1[:process_size, :, :])
        index = process_size
        while index < x_1.shape[0]:
            if index + process_size < x_1.shape[0]:
                y_1_subpart, cluster_ids_subpart = self.get_cluster_allocation(
                    x_1[index : index + process_size, :, :]
                )
            else:
                y_1_subpart, cluster_ids_subpart = self.get_cluster_allocation(
                    x_1[index : x_1.shape[0], :, :]
                )

            y_1 = np.append(y_1, y_1_subpart, axis=0)
            cluster_ids = np.append(cluster_ids, cluster_ids_subpart, axis=0)
            index = index + process_size
        df_ = pd.concat(
            [class_users, sample_users, seq_users, pd.Series(cluster_ids)],
            axis=1,
            keys=["class_id", "sample#", "seq#", "cluster_id"],
        )
        x_2 = []
        y_2_actual = []
        users = []
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
                users.append(sample_)
        y_2_actual = np.asarray(y_2_actual)
        x_transition = self.get_transition_matrix(x_2)
        return x_1, sample_users, seq_users, np.array(users), [x_transition, y_2_actual]

    def get_multiple_chunks(self, number_chunks):
        """
        Collect data from multiple data chunks
        """
        classifier_train_data_path = data_npy_folder
        data_train_list = os.listdir(classifier_train_data_path)
        chunkid = os.listdir(classifier_train_data_path)[0].split("_")[1].split(".")[0]
        _, _, _, train_users, transition_io = self.get_data(
            chunkid, classifier_train_data_path, 6000
        )
        transition_io_matrix = transition_io[0]
        y_true = transition_io[1]
        for train_chunk_index in range(1, number_chunks):
            if data_train_list[train_chunk_index].split(".")[-1] == "npy":
                chunkid = data_train_list[train_chunk_index].split("_")[1].split(".")[0]
                _, _, _, trainchunk_users, trainchunk_transition_io = self.get_data(
                    chunkid, classifier_train_data_path, 6000
                )

                train_users = np.append(train_users, trainchunk_users, axis=0)
                transition_io_matrix = np.append(
                    transition_io_matrix, trainchunk_transition_io[0], axis=0
                )
                y_true = np.append(y_true, trainchunk_transition_io[1], axis=0)
        return train_users, transition_io_matrix, y_true

    def call(self, cluster_transition_matrix):
        return self.transition_model(cluster_transition_matrix)

