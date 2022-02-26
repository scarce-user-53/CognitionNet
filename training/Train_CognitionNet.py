import pickle
import random
import sys
import time

sys.path.append("../")
import numpy as np
import pandas as pd
import tensorflow as tf
from config import *
from models.CognitionNet import *
from utils import *

tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)
tf.get_logger().setLevel("ERROR")

secondary_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.98, beta_2=0.99)
transition_model_optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.98, beta_2=0.99)


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
    Bridge_loss = 0
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

        # Reconstruct gameplay features using decoder along with context vectors
        # calculated using attention
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
            dec_input = tf.expand_dims(input_data[:, t], 1)

        # Serving K-means objective using spectral relaxation
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

        # Consider Bridge-Loss only when the transition network has learned some game dynamics
        if transition_flag:
            ls = tf.transpose(H)
            # Finding Bridge-Loss for each user
            for user_index in range(0, int(ls.shape[0]), nseq_per_user):
                ls_user = ls[user_index : user_index + nseq_per_user, :]
                CS_user = tf.matmul(ls_user, tf.transpose(ls_user))
                # Applying penalty reward to cosine similarity matrix
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
                Bridge_loss += RMSE_loss.result()

            loss = loss_1 + lamda * 0.5 * trace_loss + Bridge_loss / users_seq2seq_phase2

        else:
            loss = loss_1 + lamda * 0.5 * trace_loss

    # Updating F with some interval using SVD
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

    if transition_flag:
        return loss_1.numpy(), lamda * 0.5 * trace_loss.numpy(), Bridge_loss.numpy()
    else:
        return loss_1.numpy(), lamda * 0.5 * trace_loss.numpy(), Bridge_loss


@tf.function
def train_step_transition_model(style_model, x_transition, y):
    with tf.GradientTape() as tape:
        transition_input = x_transition.reshape(
            x_transition.shape[0], x_transition.shape[1], x_transition.shape[2], 1
        )
        y_pred, _ = style_model(transition_input)
        cce_loss = categorical_loss(y, y_pred)

    gradients = tape.gradient(cce_loss, style_model.transition_model.trainable_variables)
    transition_model_optimizer.apply_gradients(
        zip(gradients, style_model.transition_model.trainable_variables)
    )

    style_model.CCE_transition_model_independent.append(cce_loss.numpy())

    return cce_loss.numpy()


def train_transition_model(
    style_model, num_chunks=2, epochs_transition_model=64, batchsize_transition_model=32
):
    """
    Select proportionate number of users from each class in each batch
    """
    print("------Partial Training of Transition Model Start-------")

    train_users, transition_io_matrix, y_true = style_model.get_multiple_chunks(num_chunks)
    sus_index = y_true == "sustainers"
    sus_index = sus_index.reshape(sus_index.shape[0])
    sus_transition_matrix = transition_io_matrix[sus_index, :, :]

    burn_index = y_true == "burnouts"
    burn_index = burn_index.reshape(burn_index.shape[0])
    burn_transition_matrix = transition_io_matrix[burn_index, :, :]

    churn_index = y_true == "churnouts"
    churn_index = churn_index.reshape(churn_index.shape[0])
    churn_transition_matrix = transition_io_matrix[churn_index, :, :]

    no_batches = int(transition_io_matrix.shape[0] / batchsize_transition_model)

    for epoch in range(epochs_transition_model):
        epoch_cce_loss = 0
        for batch in range(no_batches):
            per_class_users = int(batchsize_transition_model / classes)

            random_choice_sustainer = random.sample(
                range(0, sus_transition_matrix.shape[0]), per_class_users
            )
            random_choice_burnouts = random.sample(
                range(0, burn_transition_matrix.shape[0]), per_class_users
            )
            random_choice_churnouts = random.sample(
                range(0, churn_transition_matrix.shape[0]),
                per_class_users + batchsize_transition_model % classes,
            )

            transition_matrix_batch = np.append(
                sus_transition_matrix[random_choice_sustainer, :, :],
                burn_transition_matrix[random_choice_burnouts, :, :],
                axis=0,
            )
            transition_matrix_batch = np.append(
                transition_matrix_batch,
                churn_transition_matrix[random_choice_churnouts, :, :],
                axis=0,
            )

            y_true_batch = np.append(
                np.asarray([["sustainers"] for i in range(per_class_users)]),
                np.asarray([["burnouts"] for i in range(per_class_users)]),
                axis=0,
            )
            y_true_batch = np.append(
                y_true_batch,
                np.asarray(
                    [
                        ["churnouts"]
                        for i in range(per_class_users + batchsize_transition_model % classes)
                    ]
                ),
            )

            cce_loss = train_step_transition_model(
                style_model,
                transition_matrix_batch,
                y_true_batch.reshape(y_true_batch.shape[0], 1),
            )
            epoch_cce_loss += cce_loss

        print("Partialy training Transition Model epoch loss:", epoch_cce_loss / no_batches)
        style_model.Sparse_CCE.append(epoch_cce_loss / no_batches)

    print("------Partial Training of Transition Model Complete-------")


def prepare_input_seq_seq_v2(x, class_users, users_list, seq_users, chunk):
    """
    Select users with continuous random gameplay sequence from each
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

    # Select random users from each class
    random_users_sustainer = random.sample(list(np.unique(sus_users)), per_class_users)
    random_users_burnouts = random.sample(list(np.unique(burn_users)), per_class_users)
    random_users_churnouts = random.sample(
        list(np.unique(churn_users)), per_class_users + users_seq2seq_phase2 % classes
    )

    # Buffer users to select when we have less sequence
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
    class_batch = []
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


def train_seq2seq(
    game_style_encoder,
    game_style_decoder,
    competitive_epoch,
    x,
    class_,
    sample_,
    seq_,
    transition_flag,
):
    """
    Training the interpretor network
    """
    game_style_encoder.transition_model.trainable = False
    for epoch in range(epochs):
        start = time.time()
        total_rcloss = 0
        total_traceloss = 0
        total_s_score = 0
        total_bridge_loss = 0

        for batch in range(no_batches):

            # Prepare coherent data for seq2seq and also for classifier model
            (
                inp,
                inp_seq_batch,
                inp_users,
                inp_batch_class,
                distinct_batch_users,
            ) = prepare_input_seq_seq_v2(x, class_, sample_, seq_, competitive_epoch)
            (
                latent_space,
                encoder_states_1,
                encoder_states_2,
                encoder_states_3,
            ) = game_style_encoder(inp)
            ls = tf.concat(
                [tf.concat([encoder_states_1[0], encoder_states_2[0]], 1), encoder_states_3[0]], 1
            )
            game_style_encoder.create_cluster(ls)
            cluster_ids = game_style_encoder.predict_cluster(ls)
            s_score = game_style_encoder.get_silhoutte_score(ls)
            trasition_model_io = game_style_encoder.transition_model.get_metadata_for_transition(
                inp_batch_class, inp_users, inp_seq_batch, cluster_ids
            )

            if trasition_model_io[-1].shape != (
                users_seq2seq_phase2,
                nseq_per_user,
                nseq_per_user,
            ):
                print(
                    inp.shape,
                    inp_seq_batch.shape,
                    inp_users.shape,
                    inp_batch_class.shape,
                    ls.numpy().shape,
                    np.asarray(cluster_ids).shape,
                    trasition_model_io[-1].shape,
                )
                print("Transition matrix", trasition_model_io[-1])

            trasition_model_io[0] = trasition_model_io[0].reshape(
                trasition_model_io[0].shape[0],
                trasition_model_io[0].shape[1],
                trasition_model_io[0].shape[2],
                1,
            )
            inp_prediction, fm_output = game_style_encoder.transition_model(trasition_model_io[0])
            cce_loss = categorical_loss(trasition_model_io[1], inp_prediction)

            start = np.zeros((inp.shape[0], inp.shape[1], 1))
            dec_input_teacher_forcing = np.append(start, inp[:, :, :-1], axis=2)

            rc_loss, trace_loss, bridge_loss = train_collaborative_step(
                inp,
                epoch,
                game_style_encoder,
                game_style_decoder,
                dec_input_teacher_forcing,
                np.asarray(trasition_model_io[-1]),
                fm_output,
                transition_flag,
            )

            total_rcloss += rc_loss
            total_traceloss += trace_loss
            total_s_score += s_score
            total_bridge_loss += bridge_loss

        print(
            "seq2seq: Epoch {} RcLoss {:.4f} TraceLoss {:.4f} Silhoutte Score {:.4f} Bridge Loss {:.4f}".format(
                epoch + 1,
                total_rcloss / no_batches,
                total_traceloss / no_batches,
                total_s_score / no_batches,
                total_bridge_loss / no_batches,
            )
        )

        game_style_encoder.TraceLoss_history.append(total_traceloss / no_batches)
        game_style_encoder.RC_history.append(total_rcloss / no_batches)
        game_style_encoder.BridgeLoss_history.append(total_bridge_loss / no_batches)
        game_style_encoder.Silhouttescore_history.append(total_s_score / no_batches)

    game_style_encoder.transition_model.trainable = True

    print("------seq2seq trained------", competitive_epoch)


def trainCognitionNet(no_collab_epochs, numchunks):
    """
    Collaborative training function
    """
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

    style_model = Play_Style_Transition(game_style_encoder, K1, classes, F_shape)
    x, class_, sample_, seq_ = data_manipulation(data_npy_folder, 1)

    for collaborative_epoch in range(0, no_collab_epochs):
        if collaborative_epoch == 0:
            train_seq2seq(
                game_style_encoder, game_style_decoder, 1, x, class_, sample_, seq_, False
            )
        else:
            train_seq2seq(
                game_style_encoder, game_style_decoder, 1, x, class_, sample_, seq_, True
            )
        game_style_encoder.transition_model.trainable = True
        train_transition_model(style_model, num_chunks=numchunks)

    print("saving model weights")
    game_style_encoder.transition_model.trainable = True
    game_style_encoder.save("../modelweights/Model_Checkpoints/Encoder_model")
    game_style_encoder.transition_model.save("../modelweights/Model_Checkpoints/Classifier_model")
    style_model.transition_model.save("../modelweights/Model_Checkpoints/Transition_model")
    with open("../modelweights/Model_Checkpoints/Kmeans_model/kmean_clustering.pkl", "wb") as f:
        pickle.dump(game_style_encoder.kmeans_cluster, f)
