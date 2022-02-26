import sys

sys.path.append("../")

# Number of collaborative epochs
collaborative_epochs = 2
# Number of users per batch
select_users_per_batch = 16
# Number of instances of continuous gameplay sequence to select for each user
seq_to_select_per_user = 4
# Dimension of penultimate layer in classification network
fm_space = 8
# Maximum number of games in a continuous gameplay sequence
window_length = 50
# Number of clusters
K1 = 7
# Feature selection
primary_list = [
    26,
    4,
    7,
    30,
    8,
    9,
    0,
    32,
    3,
    2,
    18,
    19,
    20,
    21,
    22,
    23,
    3,
    11,
    12,
    13,
]
# Number of users per batch in Bridge loss
users_seq2seq_phase2 = 15

F_shape = select_users_per_batch * seq_to_select_per_user
epochs = 12
classes = 3
nseq_per_user = 50
batch_size_seq2seq = users_seq2seq_phase2 * nseq_per_user
no_batches = 32
lamda = 0.5
data_npy_folder = "../data/train/"
data_npy_folder_test = "../data/test/"
num_features = len(primary_list)

assert users_seq2seq_phase2 % classes == 0
