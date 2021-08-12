from tokenizer import Tokenizer
from utils import load_dataset


def test_tokenizer():
    with open('res/X.csv', 'r', encoding='utf-8') as file:
        x = file.readline()
    with open('res/y_cesura.csv', 'r', encoding='utf-8') as file:
        y = file.readline()
    reserved_tokens = ['[START]', '[END]']
    tokenizer = Tokenizer(reserved_tokens, 'res/vocab.txt')
    tok_x = tokenizer.tokenize(x)
    detok_x = tokenizer.detokenize(tok_x)
    tok_y = tokenizer.tokenize(y)
    detok_y = tokenizer.detokenize(tok_y)
    assert detok_x.numpy()[0] == b'nel mezzo del cammin di nostra vita'
    assert detok_y.numpy()[0] == b'nel |mez|zo |del |cam|min $di |no|stra |vi|ta'
