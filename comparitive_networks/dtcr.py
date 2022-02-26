import pickle
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


@tf.function
def auxilary_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return bce(tf.reshape(y_true, tf.shape(y_pred)), y_pred)


class Transition_model(keras.Model):
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
        input_ = []
        sign_matrix = np.zeros((nseq_per_user, nseq_per_user))
        counter = 0
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

            if counter == 0:
                sign_matrix = np.stack((sign_matrix, user_matrix))
            else:
                sign_matrix = np.append(
                    sign_matrix,
                    user_matrix.reshape(1, user_matrix.shape[0], user_matrix.shape[1]),
                    axis=0,
                )

            counter = counter + 1

        input_ = np.asarray(input_)

        return input_, sign_matrix[1:, :, :]

    def get_metadata_for_transition(self, class_users, sample_users, seq_users, cluster_ids):

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

        # Auxiliary Classifier
        self.W_classifier_1 = Dense(16, activation="relu")
        self.W_classifier_2 = Dense(1, activation="sigmoid")

        self.TraceLoss_history = []
        self.RC_history = []
        self.DiscmLoss_history = []

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


def generate_fake_samples(batch_data):
    fake_samples = []

    for user_index in range(0, batch_data.shape[0]):
        jumbled_cols = [i for i in range(0, window_length)]
        random.shuffle(jumbled_cols)
        fake_samples.append(batch_data[user_index, jumbled_cols, :])

    fake_samples = np.asarray(fake_samples)

    assert fake_samples.shape == batch_data.shape

    return fake_samples


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


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.98, beta_2=0.99)


@tf.function
def train_collaborative_step(
    input_data,
    epoch_no,
    game_style_encoder,
    game_style_decoder,
    dec_input_teacher_forcing,
    fake_data,
):
    loss_1 = 0
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
            loss_1 += game_style_encoder.custom_loss_MSE(input_data[:, t], predictions)
            dec_input = tf.expand_dims(input_data[:, t], 1)  # Teacher Forcing

        ls_space_true = tf.concat(
            [tf.concat([encoder_states_1[0], encoder_states_2[0]], 1), encoder_states_3[0]], 1
        )

        H = tf.transpose(ls_space_true)
        FtHt = tf.matmul(tf.transpose(game_style_encoder.F), tf.transpose(H))
        HF = tf.matmul(H, game_style_encoder.F)
        trace_loss = tf.linalg.trace(tf.matmul(np.transpose(H), H)) - tf.linalg.trace(
            tf.matmul(FtHt, HF)
        )

        # Discriminator Loss
        (
            _,
            fake_encoder_states_1,
            fake_encoder_states_2,
            fake_encoder_states_3,
        ) = game_style_encoder(fake_data)
        ls_space_fake = tf.concat(
            [
                tf.concat([fake_encoder_states_1[0], fake_encoder_states_2[0]], 1),
                fake_encoder_states_3[0],
            ],
            1,
        )
        ls_space = tf.concat([ls_space_true, ls_space_fake], 0)
        aux_1 = game_style_encoder.W_classifier_1(ls_space)
        aux_pred = game_style_encoder.W_classifier_2(aux_1)
        discriminator_loss = auxilary_loss(
            aux_pred, np.append(np.ones(input_data.shape[0]), np.zeros(fake_data.shape[0]))
        )

        loss = loss_1 + lamda * 0.5 * trace_loss + tf.cast(discriminator_loss, dtype=tf.float32)

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

    return loss_1.numpy(), lamda * 0.5 * trace_loss.numpy(), discriminator_loss.numpy()


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
    for epoch in range(epochs):
        start = time.time()
        total_rcloss = 0
        total_traceloss = 0
        total_discriminator_loss = 0

        for batch in range(no_batches):
            (
                inp,
                inp_seq_batch,
                inp_users,
                inp_batch_class,
                distinct_batch_users,
            ) = prepare_input_seq_seq_v2(x, class_, sample_, seq_, competitive_epoch)

            start = np.zeros((inp.shape[0], inp.shape[1], 1))
            dec_input_teacher_forcing = np.append(start, inp[:, :, :-1], axis=2)

            inp_fake = generate_fake_samples(inp)
            rc_loss, trace_loss, discriminator_loss = train_collaborative_step(
                inp,
                epoch,
                game_style_encoder,
                game_style_decoder,
                dec_input_teacher_forcing,
                inp_fake,
            )

            total_rcloss += rc_loss
            total_traceloss += trace_loss
            total_discriminator_loss += discriminator_loss

            if batch == 0:
                print(
                    "seq2seq: Epoch {} Batch {} RC Loss {:.4f} Trace Loss {:.4f} Discriminator Loss {:.4f}".format(
                        epoch + 1, batch, rc_loss, trace_loss, discriminator_loss
                    )
                )

        print(
            "seq2seq: Epoch {} RcLoss {:.4f} TraceLoss {:.4f} DiscriminatorLoss {:.4f}".format(
                epoch + 1,
                total_rcloss / no_batches,
                total_traceloss / no_batches,
                total_discriminator_loss / no_batches,
            )
        )

        game_style_encoder.TraceLoss_history.append(total_traceloss / no_batches)
        game_style_encoder.RC_history.append(total_rcloss / no_batches)
        game_style_encoder.DiscmLoss_history.append(total_discriminator_loss / no_batches)

    print("------seq2seq trained------", competitive_epoch)


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

x, class_, sample_, seq_ = data_manipulation(data_npy_folder, 1)

for collaborative_epoch in range(0, 2):
    train_seq2seq(1, x, class_, sample_, seq_, False)


x, class_, sample_, seq_ = data_manipulation(data_npy_folder, 1)

latent_space, encoder_states_1, encoder_states_2, encoder_states_3 = game_style_encoder(x)
ls = tf.concat([tf.concat([encoder_states_1[0], encoder_states_2[0]], 1), encoder_states_3[0]], 1)

game_style_encoder.create_cluster(ls)
print("Silhoutte Score", game_style_encoder.get_silhoutte_score(ls))

with open("../modelweights/DTCR/kmean_clustering.pkl", "wb") as f:
    pickle.dump(game_style_encoder.kmeans_cluster, f)
game_style_encoder.save("../modelweights/DTCR/DTCR_Encoder_model1")

# To verify DTCR-alike results, run the following lines and feed the resultant
# csv to the corresponding ipynb notebook

# cluster_ids = game_style_encoder.predict_cluster(ls)
# cluster_results = pd.DataFrame(
#     {"user_id": sample_, "class": class_.reshape(-1), "seq": seq_, "cluster": cluster_ids}
# )
# cluster_results.to_csv("LatentSpace/Run_1/Chunk1_clusters.csv", index=False)
# np.save("LatentSpace/Run_1/Chunk1_Latent_Space.npy", ls.numpy(), allow_pickle=True)

