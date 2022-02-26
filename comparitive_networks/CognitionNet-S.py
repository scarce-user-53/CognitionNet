import random
import sys
import time

sys.path.append("../")
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from config import *
from models.CognitionNet import *
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Input
from training.Train_CognitionNet import *
from utils import *

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


class Cluster_Transition_Network(tf.keras.Model):
    def __init__(self, nseq_per_user, num_features, num_class=classes):
        super(Cluster_Transition_Network, self).__init__()
        self.nseq_per_user = nseq_per_user
        self.num_features = num_features
        self.num_class = num_class
        self.encoder_inputs = Input(
            shape=(self.nseq_per_user, self.num_features), name="encoder_input"
        )
        self.encoder_lstm_stack_1 = LSTM(64, return_sequences=True, return_state=True)

        self.W_fm = Dense(fm_space, activation="relu")
        self.W_classifier = Dense(self.num_class, activation="softmax")

    def collect_data(self, chunk_list):
        x_t, class_t, sample_t, seq_t = data_manipulation(data_npy_folder, chunk_list[0])

        for chunkidx in range(1, len(chunk_list)):
            x_check1, class_check1, sample_check1, seq_check1 = data_manipulation(
                data_npy_folder, chunk_list[chunkidx]
            )
            x_t, class_t, sample_t, seq_t = (
                np.append(x_t, x_check1, axis=0),
                np.append(class_t, class_check1, axis=0),
                np.append(sample_t, sample_check1, axis=0),
                np.append(seq_t, seq_check1, axis=0),
            )

        return x_t, class_t, sample_t, seq_t

    def get_pr_matrix(self, users, cluster, batch_users):
        pr_matrix = []

        for user in batch_users:
            pr_matrix_user = np.ones((nseq_per_user, nseq_per_user))
            mask = users == user
            cluster_user = cluster[mask]
            for i in range(0, len(cluster_user) - 1):
                for j in range(i, len(cluster_user)):
                    if cluster_user[i] == cluster_user[j]:
                        pr_matrix_user[i][j] = -1

            pr_matrix.append(pr_matrix_user)

        return np.asarray(pr_matrix)

    def call(self, input_data):
        encoder_outputs_1, state_h_1, state_c_1 = self.encoder_lstm_stack_1(input_data)
        fm_space = self.W_fm(state_h_1)
        class_prob = self.W_classifier(fm_space)
        return encoder_outputs_1, [state_h_1, state_c_1], fm_space, class_prob


@tf.function
def train_step_mini_batch_cluster_transition(batch_input, batch_class, transition_model):
    with tf.GradientTape() as tape:
        _, _, _, pred_prob = transition_model(batch_input)
        cce_loss = categorical_loss(batch_class, pred_prob)

    trainable_variables = transition_model.trainable_variables
    gradients = tape.gradient(cce_loss, trainable_variables)
    transition_model_optimizer.apply_gradients(zip(gradients, trainable_variables))
    return cce_loss


def prepare_batch_cluster_transition(x, class_users, users_list, seq_users):

    sus_index = class_users == [["sustainers"] for i in range(0, class_users.shape[0])]
    sus_index = sus_index.reshape(sus_index.shape[0])
    sus_x = x[sus_index, :, :]
    sus_users = np.asarray(users_list[sus_index])
    sus_seq = seq_users[sus_index]

    burn_index = class_users == [["burnouts"] for i in range(0, class_users.shape[0])]
    burn_index = burn_index.reshape(burn_index.shape[0])
    burn_x = x[burn_index, :, :]
    burn_users = np.asarray(users_list[burn_index])
    burn_seq = seq_users[burn_index]

    churn_index = class_users == [["churnouts"] for i in range(0, class_users.shape[0])]
    churn_index = churn_index.reshape(churn_index.shape[0])
    churn_x = x[churn_index, :, :]
    churn_users = np.asarray(users_list[churn_index])
    churn_seq = seq_users[churn_index]

    per_class_users = int(users_seq2seq_phase2 / classes)

    # Get random users from each class
    random_users_sustainer = random.sample(list(np.unique(sus_users)), per_class_users)
    random_users_burnouts = random.sample(list(np.unique(burn_users)), per_class_users)
    random_users_churnouts = random.sample(list(np.unique(churn_users)), per_class_users)

    chunk_buffer = pd.DataFrame(
        {
            "class": class_users.reshape(-1),
            "sample": users_list.reshape(-1),
            "seq_length": np.ones((users_list.shape[0])),
        }
    )
    chunk_buffer = (
        chunk_buffer.groupby(by=["class", "sample"]).agg({"seq_length": "sum"}).reset_index()
    )
    chunk_buffer = chunk_buffer[chunk_buffer["seq_length"] > nseq_per_user]

    x_batch = []
    seq_batch = []
    users_batch = []
    class_users = []

    users_batch_distinct = []
    users_batch_distinct.extend(random_users_sustainer)
    users_batch_distinct.extend(random_users_burnouts)
    users_batch_distinct.extend(random_users_churnouts)

    for user in random_users_sustainer:
        indices = sus_users == [user for i in range(0, sus_users.shape[0])]
        x_subset = sus_x[indices, :, :]
        seq_subset = sus_seq[indices]
        class_users.extend([["sustainers"]])

        if x_subset.shape[0] - nseq_per_user > 0:
            end_seq = nseq_per_user
            users_batch.extend(np.asarray([user for i in range(0, nseq_per_user)]))
        else:
            end_seq = x_subset.shape[0]
            users_batch.extend(np.asarray([user for i in range(0, x_subset.shape[0])]))

        if len(x_batch) == 0:
            x_batch = x_subset[0:end_seq, :, :]
            seq_batch = seq_subset[0:end_seq]
        else:
            x_batch = np.append(x_batch, x_subset[0:end_seq, :, :], axis=0)
            seq_batch = np.append(seq_batch, seq_subset[0:end_seq], axis=0)

    for user in random_users_burnouts:
        indices = burn_users == [user for i in range(0, burn_users.shape[0])]
        x_subset = burn_x[indices, :, :]
        seq_subset = burn_seq[indices]
        class_users.extend([["burnouts"]])

        if x_subset.shape[0] - nseq_per_user > 0:
            end_seq = nseq_per_user
            users_batch.extend(np.asarray([user for i in range(0, nseq_per_user)]))
        else:
            end_seq = x_subset.shape[0]
            users_batch.extend(np.asarray([user for i in range(0, x_subset.shape[0])]))

        if len(x_batch) == 0:
            x_batch = x_subset[0:end_seq, :, :]
            seq_batch = seq_subset[0:end_seq]
        else:
            x_batch = np.append(x_batch, x_subset[0:end_seq, :, :], axis=0)
            seq_batch = np.append(seq_batch, seq_subset[0:end_seq], axis=0)

    for user in random_users_churnouts:
        indices = churn_users == [user for i in range(0, churn_users.shape[0])]
        x_subset = churn_x[indices, :, :]
        seq_subset = churn_seq[indices]
        class_users.extend([["churnouts"]])

        if x_subset.shape[0] - nseq_per_user > 0:
            end_seq = nseq_per_user
            users_batch.extend(np.asarray([user for i in range(0, nseq_per_user)]))
        else:
            end_seq = x_subset.shape[0]
            users_batch.extend(np.asarray([user for i in range(0, x_subset.shape[0])]))

        if len(x_batch) == 0:
            x_batch = x_subset[0:end_seq, :, :]
            seq_batch = seq_subset[0:end_seq]
        else:
            x_batch = np.append(x_batch, x_subset[0:end_seq, :, :], axis=0)
            seq_batch = np.append(seq_batch, seq_subset[0:end_seq], axis=0)

    return (
        np.asarray(x_batch),
        np.asarray(seq_batch),
        np.asarray(class_users),
        np.asarray(users_batch_distinct),
        np.asarray(users_batch),
    )


def train_cluster_transition_network(
    x_cluster, class_cluster, sample_cluster, seq_cluster, transition_model
):
    for epoch in range(8):
        epoch_cce_loss = 0

        for batch in range(no_batches):
            (
                x_batch,
                seq_batch,
                class_users,
                users_batch_distinct,
                users_list,
            ) = prepare_batch_cluster_transition(
                x_cluster, class_cluster, sample_cluster, seq_cluster
            )
            assert x_batch.shape[0] == users_list.shape[0]

            enc_output, _h1, _h2, _h3 = game_style_encoder(x_batch)
            ls_batch = tf.concat([tf.concat([_h1[0], _h2[0]], 1), _h3[0]], 1)

            assert ls_batch.shape[0] == x_batch.shape[0]
            input_cluster_transition = []

            for user in range(0, users_batch_distinct.shape[0]):
                mask = users_list == users_batch_distinct[user]
                mask = np.asarray(mask)
                user_ls_seq = tf.boolean_mask(ls_batch, mask)
                user_ls_seq = user_ls_seq.numpy()

                if user_ls_seq.shape[0] < nseq_per_user:
                    deficit_seq = int(nseq_per_user - user_ls_seq.shape[0])
                    user_ls_seq = np.append(
                        user_ls_seq, np.zeros((deficit_seq, user_ls_seq.shape[1])), axis=0
                    )

                input_cluster_transition.append(user_ls_seq[:nseq_per_user, :])

            # Call train_step on mini batch
            input_cluster_transition = np.asarray(input_cluster_transition)
            batch_cce_loss = train_step_mini_batch_cluster_transition(
                input_cluster_transition, class_users, transition_model
            )
            epoch_cce_loss += batch_cce_loss

        print("Epoch CCE loss: ", epoch_cce_loss.numpy())


class Game_Behavior_Encoder(keras.Model):
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
        self.kmeans_cluster = KMeans(n_clusters=self.K1, random_state=0)

        self.transition_model = Cluster_Transition_Network(nseq_per_user, 64 + 32 + num_features)

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

        self.W_b = Dense(1, activation="sigmoid")
        self.W_c = Dense(fm_space, activation="relu")

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
        print("Starting to Prepare Cluster")
        self.kmeans_cluster.fit(latent_space)

    def predict_cluster(self, latent_space):
        return self.kmeans_cluster.predict(latent_space.numpy())


class Game_Behavior_Decoder(keras.Model):
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


def data_manipulation(data_folder, data_file_no):

    data_file = data_folder + "chunk_" + str(data_file_no) + ".npy"
    data_chunk = np.load(data_file, allow_pickle=True)

    x = data_chunk[:, :window_length, :-7]

    # Using primary data columns
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


@tf.function
def train_collaborative_step(
    input_data,
    epoch_no,
    game_style_encoder,
    game_style_decoder,
    dec_input_teacher_forcing,
    pr_sign_matrix,
    feature_output,
    transition_flag,
):
    loss_1 = 0
    FCM_loss = 0
    with tf.GradientTape() as tape:

        latent_space, encoder_states_1, encoder_states_2, encoder_states_3 = game_style_encoder(
            input_data
        )
        dec_input = tf.expand_dims(
            [np.zeros(game_style_encoder.num_features)] * input_data.shape[0], 1
        )

        decoder_states_1, decoder_states_2, decoder_states_3 = (
            encoder_states_1,
            encoder_states_2,
            encoder_states_3,
        )

        for t in range(0, game_style_encoder.window_length):
            (
                predictions,
                decoder_states_1,
                decoder_states_2,
                decoder_states_3,
                attention_weights,
            ) = game_style_decoder(
                dec_input, decoder_states_1, decoder_states_2, decoder_states_3, latent_space
            )
            loss_1 += game_style_encoder.custom_loss_MSE(
                input_data[:, t], predictions
            )  # Teacher Forcing
            dec_input = tf.expand_dims(input_data[:, t], 1)

        H = tf.transpose(
            tf.concat(
                [tf.concat([encoder_states_1[0], encoder_states_2[0]], 1), encoder_states_3[0]], 1
            )
        )
        FtHt = tf.matmul(tf.transpose(game_style_encoder.F), tf.transpose(H))
        HF = tf.matmul(H, game_style_encoder.F)
        trace_loss = tf.linalg.trace(tf.matmul(np.transpose(H), H)) - tf.linalg.trace(
            tf.matmul(FtHt, HF)
        )
        loss = loss_1 + lamda * 0.5 * trace_loss

        if transition_flag:
            ls = tf.transpose(H)
            for user_index in range(0, int(ls.shape[0]), nseq_per_user):
                ls_user = ls[user_index : user_index + nseq_per_user, :]
                CS_user = tf.matmul(ls_user, tf.transpose(ls_user))
                PR_similarity_user = tf.multiply(
                    CS_user, pr_sign_matrix[int(user_index / nseq_per_user), :, :]
                )

                assert CS_user.shape == (nseq_per_user, nseq_per_user)
                assert pr_sign_matrix.shape == (users_seq2seq_phase2, nseq_per_user, nseq_per_user)

                feature_generated = game_style_encoder.W_b(PR_similarity_user)
                feature_generated = tf.reshape(feature_generated, [1, feature_generated.shape[0]])
                feature_generated = game_style_encoder.W_c(feature_generated)
                feature_generated = tf.reshape(feature_generated, feature_generated.shape[1])

                RMSE_loss = tf.keras.metrics.RootMeanSquaredError()
                RMSE_loss.update_state(
                    feature_output[int(user_index / nseq_per_user), :], feature_generated
                )
                FCM_loss += RMSE_loss.result()

            loss = loss_1 + lamda * 0.5 * trace_loss + FCM_loss / users_seq2seq_phase2

        else:
            loss = loss_1 + lamda * 0.5 * trace_loss

    if (epoch_no % 10 == 0) & (epoch_no != 0):
        U, sigma, VT = np.linalg.svd(H)
        sorted_indices = np.argsort(sigma)
        topk_evecs = VT[sorted_indices[: -K1 - 1 : -1], :]
        game_style_encoder.F = tf.Variable(np.transpose(topk_evecs))

    trainable_variables = (
        game_style_encoder.trainable_variables + game_style_decoder.trainable_variables
    )
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss_1, lamda * 0.5 * trace_loss, FCM_loss


def prepare_input_seq_seq_v2(x, class_users, users_list, seq_users, chunk):
    """
    Function to select continuous sequences
    """
    sus_index = class_users == [["sustainers"] for i in range(0, class_users.shape[0])]
    sus_index = sus_index.reshape(sus_index.shape[0])
    sus_x = x[sus_index, :, :]
    sus_users = np.asarray(users_list[sus_index])
    sus_seq = seq_users[sus_index]

    burn_index = class_users == [["burnouts"] for i in range(0, class_users.shape[0])]
    burn_index = burn_index.reshape(burn_index.shape[0])
    burn_x = x[burn_index, :, :]
    burn_users = np.asarray(users_list[burn_index])
    burn_seq = seq_users[burn_index]

    churn_index = class_users == [["churnouts"] for i in range(0, class_users.shape[0])]
    churn_index = churn_index.reshape(churn_index.shape[0])
    churn_x = x[churn_index, :, :]
    churn_users = np.asarray(users_list[churn_index])
    churn_seq = seq_users[churn_index]

    per_class_users = int(users_seq2seq_phase2 / classes)

    # Get random users from each class
    random_users_sustainer = random.sample(list(np.unique(sus_users)), per_class_users)
    random_users_burnouts = random.sample(list(np.unique(burn_users)), per_class_users)
    random_users_churnouts = random.sample(
        list(np.unique(churn_users)), per_class_users + users_seq2seq_phase2 % classes
    )

    chunk_buffer = pd.DataFrame(
        {
            "class": class_users.reshape(-1),
            "sample": users_list.reshape(-1),
            "seq_length": np.ones((users_list.shape[0])),
        }
    )
    chunk_buffer = (
        chunk_buffer.groupby(by=["class", "sample"]).agg({"seq_length": "sum"}).reset_index()
    )
    chunk_buffer = chunk_buffer[chunk_buffer["seq_length"] > nseq_per_user]

    x_batch = []
    seq_batch = []
    users_batch = []
    users_batch_distinct = []
    users_batch_distinct.extend(random_users_sustainer)
    users_batch_distinct.extend(random_users_burnouts)
    users_batch_distinct.extend(random_users_churnouts)

    for user in random_users_sustainer:
        indices = sus_users == [user for i in range(0, sus_users.shape[0])]
        x_subset = sus_x[indices, :, :]
        seq_subset = sus_seq[indices]

        if x_subset.shape[0] - nseq_per_user > 0:

            users_batch.extend(np.asarray([user for i in range(0, nseq_per_user)]))

            random_index_start = int(
                random.sample(range(0, x_subset.shape[0] - nseq_per_user), 1)[0]
            )

            if len(x_batch) == 0:
                x_batch = x_subset[random_index_start : random_index_start + nseq_per_user, :, :]
                seq_batch = seq_subset[random_index_start : random_index_start + nseq_per_user]
            else:
                x_batch = np.append(
                    x_batch,
                    x_subset[random_index_start : random_index_start + nseq_per_user, :, :],
                    axis=0,
                )
                seq_batch = np.append(
                    seq_batch,
                    seq_subset[random_index_start : random_index_start + nseq_per_user],
                    axis=0,
                )
        else:
            chunk_buffer_sus = chunk_buffer[
                (chunk_buffer["class"] == "sustainers")
                & (~chunk_buffer["sample"].isin(random_users_sustainer))
            ]
            buffer_sus = list(chunk_buffer_sus["sample"].sample(n=1))
            random_users_sustainer.extend(buffer_sus)

    for user in random_users_burnouts:
        indices = burn_users == [user for i in range(0, burn_users.shape[0])]

        x_subset = burn_x[indices, :, :]
        seq_subset = burn_seq[indices]

        if x_subset.shape[0] - nseq_per_user > 0:
            users_batch.extend(np.asarray([user for i in range(0, nseq_per_user)]))
            random_index_start = int(
                random.sample(range(0, x_subset.shape[0] - nseq_per_user), 1)[0]
            )

            if len(x_batch) == 0:
                x_batch = x_subset[random_index_start : random_index_start + nseq_per_user, :, :]
                seq_batch = seq_subset[random_index_start : random_index_start + nseq_per_user]
            else:
                x_batch = np.append(
                    x_batch,
                    x_subset[random_index_start : random_index_start + nseq_per_user, :, :],
                    axis=0,
                )
                seq_batch = np.append(
                    seq_batch,
                    seq_subset[random_index_start : random_index_start + nseq_per_user],
                    axis=0,
                )
        else:
            chunk_buffer_burn = chunk_buffer[
                (chunk_buffer["class"] == "burnouts")
                & (~chunk_buffer["sample"].isin(random_users_burnouts))
            ]
            buffer_burn = list(chunk_buffer_burn["sample"].sample(n=1))
            random_users_burnouts.extend(buffer_burn)

    for user in random_users_churnouts:
        indices = churn_users == [user for i in range(0, churn_users.shape[0])]

        x_subset = churn_x[indices, :, :]
        seq_subset = churn_seq[indices]

        if x_subset.shape[0] - nseq_per_user > 0:
            users_batch.extend(np.asarray([user for i in range(0, nseq_per_user)]))
            random_index_start = int(
                random.sample(range(0, x_subset.shape[0] - nseq_per_user), 1)[0]
            )

            if len(x_batch) == 0:
                x_batch = x_subset[random_index_start : random_index_start + nseq_per_user, :, :]
                seq_batch = seq_subset[random_index_start : random_index_start + nseq_per_user]
            else:
                x_batch = np.append(
                    x_batch,
                    x_subset[random_index_start : random_index_start + nseq_per_user, :, :],
                    axis=0,
                )
                seq_batch = np.append(
                    seq_batch,
                    seq_subset[random_index_start : random_index_start + nseq_per_user],
                    axis=0,
                )
        else:
            chunk_buffer_churn = chunk_buffer[
                (chunk_buffer["class"] == "churnouts")
                & (~chunk_buffer["sample"].isin(random_users_churnouts))
            ]
            buffer_churn = list(chunk_buffer_churn["sample"].sample(n=1))
            random_users_churnouts.extend(buffer_churn)

    class_s = np.asarray([["sustainers"] for i in range(0, per_class_users * nseq_per_user)])
    class_b = np.asarray([["burnouts"] for i in range(0, per_class_users * nseq_per_user)])
    class_c = np.asarray([["churnouts"] for i in range(0, per_class_users * nseq_per_user)])
    batch_class = np.append(np.append(class_s, class_b, axis=0), class_c, axis=0)
    return (
        np.asarray(x_batch),
        np.asarray(seq_batch),
        np.asarray(users_batch),
        np.asarray(batch_class),
        np.asarray(users_batch_distinct),
    )


def train_seq2seq(competitive_epoch, x, class_, sample_, seq_, transition_flag):
    game_style_encoder.transition_model.trainable = False
    process_size = 6000

    # Create cluster
    _, _h1l, _h2l, _h3l = game_style_encoder(x[:process_size, :, :])
    _h10 = _h1l[0]
    _h20 = _h2l[0]
    _h30 = _h3l[0]

    index = process_size

    while index < x.shape[0]:

        if index + process_size < x.shape[0]:
            _, sub_h1, sub_h2, sub_h3 = game_style_encoder(x[index : index + process_size, :, :])

        else:
            _, sub_h1, sub_h2, sub_h3 = game_style_encoder(x[index : x.shape[0], :, :])

        _h10 = np.append(_h10, sub_h1[0], axis=0)
        _h20 = np.append(_h20, sub_h2[0], axis=0)
        _h30 = np.append(_h30, sub_h3[0], axis=0)

        index = index + process_size

    ls = tf.concat([tf.concat([_h10, _h20], 1), _h30], 1)
    game_style_encoder.create_cluster(ls.numpy())
    print("clustering complete")

    for epoch in range(epochs):
        start = time.time()
        total_rcloss = 0
        total_traceloss = 0
        total_fcm_loss = 0

        for batch in range(no_batches):
            # Prepare coherant data for seq2seq and also for convolution model
            (
                inp,
                inp_seq_batch,
                inp_users,
                inp_batch_class,
                distinct_batch_users,
            ) = prepare_input_seq_seq_v2(x, class_, sample_, seq_, competitive_epoch)

            _, _h1, _h2, _h3 = game_style_encoder(inp)
            ls_batch = tf.concat([tf.concat([_h1[0], _h2[0]], 1), _h3[0]], 1)

            cluster_ids = game_style_encoder.predict_cluster(ls_batch)

            assert ls_batch.shape[0] == inp.shape[0]
            input_cluster_transition = []

            for user in distinct_batch_users:
                mask = inp_users == user
                mask = np.asarray(mask)
                user_ls_seq = tf.boolean_mask(ls_batch, mask)
                user_ls_seq = user_ls_seq.numpy()

                if user_ls_seq.shape[0] < nseq_per_user:
                    deficit_seq = int(nseq_per_user - user_ls_seq.shape[0])
                    user_ls_seq = np.append(
                        user_ls_seq, np.zeros((deficit_seq, user_ls_seq.shape[1])), axis=0
                    )

                input_cluster_transition.append(user_ls_seq[:nseq_per_user, :])

            input_cluster_transition = np.asarray(input_cluster_transition)
            _, _, feature_output, pred_prob = game_style_encoder.transition_model(
                input_cluster_transition
            )

            batch_pr_matrix = game_style_encoder.transition_model.get_pr_matrix(
                inp_users, cluster_ids, distinct_batch_users
            )

            start = np.zeros((inp.shape[0], inp.shape[1], 1))
            dec_input_teacher_forcing = np.append(start, inp[:, :, :-1], axis=2)

            rc_loss, trace_loss, fcm_loss = train_collaborative_step(
                inp,
                epoch,
                game_style_encoder,
                game_style_decoder,
                dec_input_teacher_forcing,
                batch_pr_matrix,
                feature_output,
                transition_flag,
            )

            total_rcloss += rc_loss
            total_traceloss += trace_loss
            total_fcm_loss += fcm_loss

        print(
            "seq2seq: Epoch {} RcLoss {:.4f} TraceLoss {:.4f} FCMLoss {:.4f}".format(
                epoch + 1,
                total_rcloss / no_batches,
                total_traceloss / no_batches,
                total_fcm_loss / no_batches,
            )
        )

    game_style_encoder.transition_model.trainable = True

    print("------seq2seq trained------", competitive_epoch)


def prediction(encoder, loc, chunkno=2):
    """
    Use this function for inferencing using CognitionNet-S model.

    Parameters
    ----------
    encoder
        Encoder object
    loc
        Test data location (same as that for CognitionNet-TM)
    chunkno, optional
        Data chunk number, by default 2
    """
    x_inf, class_inf, sample_inf, _ = data_manipulation(loc, chunkno)

    _, _h1, _h2, _h3 = encoder(x_inf)
    ls_batch = tf.concat([tf.concat([_h1[0], _h2[0]], 1), _h3[0]], 1)

    users_inf = []
    user_class_inf = []
    input_cluster_transition = []
    all_cluster = []

    cluster_ids = game_style_encoder.predict_cluster(ls_batch)

    for user in np.unique(sample_inf):
        users_inf.append(user)
        mask = sample_inf == user
        mask = np.asarray(mask)

        user_cluster = cluster_ids[mask]

        all_cluster.append(user_cluster)
        user_class_inf.append(class_inf[mask][0])

        user_ls_seq = tf.boolean_mask(ls_batch, mask)
        user_ls_seq = user_ls_seq.numpy()
        if user_ls_seq.shape[0] < nseq_per_user:
            deficit_seq = int(nseq_per_user - user_ls_seq.shape[0])
            user_ls_seq = np.append(
                user_ls_seq, np.zeros((deficit_seq, user_ls_seq.shape[1])), axis=0
            )

        input_cluster_transition.append(user_ls_seq[:nseq_per_user, :])

    _, _, _, prediction = encoder.transition_model(np.asarray(input_cluster_transition))
    y_pred_inf = np.argmax(prediction.numpy(), axis=1)
    mapping = {"sustainers": 0, "burnouts": 1, "churnouts": 2}
    y_true_inf = [mapping[item[0]] for item in user_class_inf]

    print(confusion_matrix(y_true_inf, y_pred_inf))

    return pd.DataFrame(
        {
            "user_id": users_inf,
            "prediction": y_pred_inf,
            "class": y_true_inf,
            "cluster": all_cluster,
        }
    )


# Train the Network
transition_model_optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.98, beta_2=0.99)

game_style_encoder = Game_Behavior_Encoder(
    num_class=classes,
    window_length=window_length,
    num_features=num_features,
    k=K1,
    latent_dim=num_features - 2,
)
game_style_decoder = Game_Behavior_Decoder(
    num_class=classes,
    window_length=window_length,
    num_features=num_features,
    k=K1,
    latent_dim=num_features - 2,
)

x_total, class_total, sample_total, seq_total = game_style_encoder.transition_model.collect_data(
    [1]
)

for collaborative_epoch in range(0, 2):
    x, class_, sample_, seq_ = data_manipulation(data_npy_folder, 1)
    print("training collaborative_epoch:", collaborative_epoch)

    if collaborative_epoch == 0:
        train_seq2seq(1, x, class_, sample_, seq_, False)
    else:
        train_seq2seq(1, x, class_, sample_, seq_, True)

    train_cluster_transition_network(
        x_total, class_total, sample_total, seq_total, game_style_encoder.transition_model
    )

# Save models post training the CognitionNet-S model, to be used further for inferencing
game_style_encoder.transition_model.trainable = True
game_style_encoder.save("../model_weights/Cognition_S/Encoder_model")
game_style_encoder.transition_model.save("../model_weights/Cognition_S/Classifier_model")

