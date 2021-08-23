# Code mainly from Tensorflow guide (https://www.tensorflow.org/text/guide/subwords_tokenizer)

import tensorflow as tf
import tensorflow_text as text
import pathlib
import re


def add_start_end(ragged, reserved_tokens):
    start = tf.argmax(tf.constant(reserved_tokens) == "[START]")
    end = tf.argmax(tf.constant(reserved_tokens) == "[END]")
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], start)
    ends = tf.fill([count, 1], end)
    return tf.concat([starts, ragged, ends], axis=1)


def cleanup_text(reserved_tokens, token_txt):
    # Drop the reserved tokens, except for "[C] and [SYL]".
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok not in ["C", "S", "Y", "N"]]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    # Join them into strings.
    result = tf.strings.reduce_join(result, separator='', axis=-1)
    result = tf.strings.regex_replace(result, 'C', '$')
    result = tf.strings.regex_replace(result, 'Y', '|')
    result = tf.strings.regex_replace(result, 'S', ' ')
    result = tf.strings.regex_replace(result, 'N', '\n')

    return result


class Tokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        super().__init__()
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=False)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        enc = add_start_end(enc, self._reserved_tokens)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)
