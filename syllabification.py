import numpy as np
import tensorflow as tf
import tensorflow_text as text
from sklearn.model_selection import train_test_split
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from tokenizer import Tokenizer

def load_dataset():
    with open('res/X.csv', 'r+', encoding='utf-8') as file:
        X = file.readlines()
    with open('res/y_cesura.csv', 'r+', encoding='utf-8') as file:
        y = file.readlines()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.03, random_state=42)
    train = list(zip(X_train, y_train))
    train = tf.data.Dataset.from_tensor_slices(train)
    val = list(zip(X_val, y_val))
    val = tf.data.Dataset.from_tensor_slices(val)
    dataset = {}
    dataset['train'] = train
    dataset['val'] = val
    return dataset



def main():
    dataset = load_dataset()
    train_dataset = dataset['train']
    X_train = train_dataset.map(lambda x: x[0])
    y_train = train_dataset.map(lambda x: x[1])
    tokenizer = Tokenizer(reserved_tokens=['[START]', '[END]', '[SYL]', '[C]'], vocab_path='res/vocab.txt')
    for examples in X_train.batch(3).take(1):
        for ex in examples:
            print(ex.numpy())
        token_batch = tokenizer.tokenize(examples)
        for ex in token_batch.to_list():
            print(ex)
        txt_tokens = tokenizer.detokenize(token_batch)
        print(txt_tokens)


if __name__ == '__main__':
    main()
