import numpy as np
import tensorflow as tf
import tensorflow_text as text
from sklearn.model_selection import train_test_split
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from tokenizer import Tokenizer
from matplotlib import pyplot as plt
from preprocessing import TRIPLET


def _train_val_test_split(X, y, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.02, random_state=random_state)
    train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset = {'train': train, 'val': val, 'test': test}
    return dataset


def load_dataset():
    with open('res/X.csv', 'r+', encoding='utf-8') as file:
        X = file.readlines()
    with open('res/y_cesura.csv', 'r+', encoding='utf-8') as file:
        y = file.readlines()
    return _train_val_test_split(X, y, random_state=42)


def load_dataset_for_gen():
    with open('res/X_gen.csv', 'r+', encoding='utf-8') as file:
        text = file.readlines()
    text = ''.join(text)
    triplets = text.split(f'{TRIPLET}')
    X = triplets[:-1]
    y = triplets[1:]
    return  _train_val_test_split(X, y, random_state=42)

def get_angles(pos, _2i, d_model):
    # 2* _2i//2 returns 2i in both cases the arg is 2i or 2i+1
    return pos / np.power(10000, (2 * _2i // 2 / np.float32(d_model)))


def positional_encoding(pos, d_model):
    angle_rads = get_angles(np.arange(pos)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices of the array: 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices of the array: 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding to attention logits
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    lower_triangular_part = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    mask = 1 - lower_triangular_part
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    qk = tf.matmul(q, k, transpose_b=True)
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = qk / tf.math.sqrt(d_k)
    # Add the mask
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    result = tf.matmul(attention_weights, v)
    return result, attention_weights


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def plot_accuracy(train_losses, train_accuracies, val_losses, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ax1.set_title('Accuracy')
    ax1.set(xlabel='Epoch')
    ax1.plot(train_accuracies, label='Train')
    ax1.plot(val_accuracies, label='Validation')
    start, end = ax1.get_xlim()
    ax1.set_xticks(np.arange(start, end, 1))

    ax2.set_title('Loss')
    ax2.set(xlabel='Epoch')
    ax2.plot(train_losses, label='Train')
    ax2.plot(val_losses, label='Validation')
    start, end = ax2.get_xlim()
    ax2.set_xticks(np.arange(start, end, 1))
    plt.show()


if __name__ == '__main__':
    load_dataset_for_gen()