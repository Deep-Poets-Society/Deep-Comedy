import re
import pickle
import string
import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

CESURA = "C"
SYL = "S"


def preprocess():
    with open('res/divina_commedia_raw.txt', 'r+', encoding='utf-8') as file:
        raw_text = ''.join(file.readlines())
    # remove capital letters
    processed_text = raw_text.lower()
    # convert three different type of ticks in ’:
    processed_text = re.sub(r'[’‘\']', '’', processed_text)
    # remove punctuation, but ticks
    processed_text = re.sub(r'[,.;:\-—?!"()\[\]«»“‟”]', '', processed_text)
    # remove cantos' headings
    processed_text = re.sub(r'.* • canto .*', '', processed_text)
    # remove lines' numbers
    processed_text = re.sub(r'\n *\d* ', '\n', processed_text)
    # remove multiple blank lines
    processed_text = re.sub(r'\n+', '\n', processed_text)
    # remove initial and last blank lines
    processed_text = re.sub(r'(^\n)|(\n$)', '', processed_text)
    # remove multiple spaces and transform spaces in [S]
    processed_text = re.sub(r' +', ' ', processed_text)

    # add <syl> token in y.csv, remove syllabification in X.csv
    x_text = re.sub(r'\|', '', processed_text)
    y_text = re.sub(r'\|', r'S', processed_text)
    # remove spaces at the beginning of each line in y.csv
    y_text = re.sub(r'^ ', '', y_text)
    # generate y_cesura.csv
    y_cesura_text = cesura(x_text, y_text)

    # save files
    with open('res/X.csv', 'w+', encoding='utf-8') as file:
        file.writelines(x_text)
    with open('res/y.csv', 'w+', encoding='utf-8') as file:
        file.writelines(y_text)
    with open('res/y_cesura.csv', 'w+', encoding='utf-8') as file:
        file.writelines(y_cesura_text)

    text_no_tag = re.sub(r'(\[START] )|( \[END])|(\[S] )|(S)|(C)', '', y_text)
    create_vocabulary(text_no_tag)


def cesura(text, syl_text):
    dictionary = pickle.load(open('res/dantes_dictionary.pkl', 'rb'))
    # add entries not present
    dictionary.update({'aegypto': [((0, 3, -1, 0), 'ae|gyp|to', 1)]})
    for word in ['te', 'beati', 'qui', 'neque', 'labïa', 'gloria']:
        dictionary[f'’{word}'] = dictionary[f'‘{word}']
    text = re.sub(r'(\[START] )|( \[END])', '', text)
    syl_text = re.sub(r'(\[START] S )|( \[END])', '', syl_text)
    lines = text.split('\n')
    syl_lines = syl_text.split('\n')
    for i, (line, syl_line) in enumerate(zip(lines, syl_lines)):
        tonic_accents = []
        counter = 0
        words = line.split(' ')
        for w in words:
            num_syl = dictionary[w][0][0][1]
            pos_accent = dictionary[w][0][0][2]
            tonic_accents += [False for _ in range(num_syl)]
            tonic_accents[pos_accent - 1] = True
        syllables = syl_line.split('S')
        syllables.remove('')
        if len(syllables) != len(tonic_accents):
            for j, s in enumerate(syllables):
                for _ in re.findall(r' .', s):
                    del tonic_accents[j]
        # hendecasyllable a maiore, cesura maschile
        elif tonic_accents[5] and syllables[5].endswith(' '):
            line_w_cesura = 'S'.join(syllables[0:6]) + 'C' + 'S'.join(syllables[6:])
        # hendecasyllable a maiore, cesura femminile
        elif tonic_accents[5] and syllables[6].endswith(' '):
            line_w_cesura = 'S'.join(syllables[0:7]) + 'C' + 'S'.join(syllables[7:])
        # hendecasyllable a minore, cesura maschile
        elif tonic_accents[3] and syllables[3].endswith(' '):
            line_w_cesura = 'S'.join(syllables[0:4]) + 'C' + 'S'.join(syllables[4:])
        # hendecasyllable a minore, cesura femminile
        elif tonic_accents[3] and syllables[4].endswith(' '):
            line_w_cesura = 'S'.join(syllables[0:5]) + 'C' + 'S'.join(syllables[5:])
        # cesura lirica
        if tonic_accents[2] and tonic_accents[5] and syllables[3].endswith(' '):
            line_w_cesura = 'S'.join(syllables[0:4]) + 'C' + 'S'.join(syllables[4:])
        # hendecasyllable a maiore, cesura after a proparoxytone
        elif tonic_accents[5] and syllables[7].endswith(' '):
            line_w_cesura = 'S'.join(syllables[0:8]) + 'C' + 'S'.join(syllables[8:])
        # hendecasyllable a minore, cesura after a proparoxytone
        elif tonic_accents[3] and syllables[5].endswith(' '):
            line_w_cesura = 'S'.join(syllables[0:6]) + 'C' + 'S'.join(syllables[6:])
        # hendecasyllable a maiore with cesura femminile between two words joint by sinalefe
        elif tonic_accents[5] and ' ' in syllables[6]:
            line_w_cesura = 'S'.join(syllables[0:6]) + 'S' + \
                            re.sub(r' (?=.)', r' C', syllables[6]) + \
                            'S' + 'S'.join(syllables[7:])
        # hendecasyllable a minore with cesura femminile between two words joint by sinalefe
        elif tonic_accents[3] and ' ' in syllables[4]:
            line_w_cesura = 'S'.join(syllables[0:4]) + 'S' + \
                            re.sub(r' (?=.)', r' C', syllables[4]) + \
                            'S' + 'S'.join(syllables[5:])
        # hendecasyllable a maiore with cesura after a proparoxytone and between two words joint by sinalefe
        elif tonic_accents[5] and ' ' in syllables[7]:
            line_w_cesura = 'S'.join(syllables[0:7]) + 'S' + \
                            re.sub(r' (?=.)', r' C', syllables[7]) + \
                            'S' + 'S'.join(syllables[8:])
        # hendecasyllable a minore with cesura after a proparoxytone and between two words joint by sinalefe
        elif tonic_accents[3] and ' ' in syllables[5]:
            line_w_cesura = 'S'.join(syllables[0:5]) + 'S' + \
                            re.sub(r' (?=.)', r' C', syllables[5]) + \
                            'S' + 'S'.join(syllables[6:])
        # There are 4 hendecasyllable a maiore where the cesura split a compound word which ends with -mente (3
        # cases) or -zial (1 case)
        elif 'mente ' == syllables[7] + syllables[8]:
            line_w_cesura = 'S'.join(syllables[0:7]) + 'C' + 'S'.join(syllables[7:])
        elif 'zial ' == syllables[6] + syllables[7]:
            line_w_cesura = 'S'.join(syllables[0:6]) + 'C' + 'S'.join(syllables[6:])
        # Remaining 7 verses are non-canonical hendecasyllables so they do not have a cesura
        else:
            line_w_cesura = syl_line
        # remove eventual multiple spaces
        line_w_cesura = re.sub(' +', ' ', line_w_cesura)
        syl_lines[i] = line_w_cesura
    return '\n'.join(syl_lines)


def create_vocabulary(text):
    y = tf.data.Dataset.from_tensor_slices(text.split('\n'))
    tokenizer_params = dict(lower_case=False)
    reserved_tokens = ['[START]', '[END]', 'S', 'C', '##S', '##C']
    vocab_args = dict(
        # The target vocabulary size
        vocab_size=200,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params=tokenizer_params,
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )
    vocab = bert_vocab.bert_vocab_from_dataset(
        y.batch(1000).prefetch(2),
        **vocab_args
    )
    with open('res/vocab.txt', 'w', encoding='utf-8') as f:
        for token in vocab:
            print(token, file=f)


if __name__ == '__main__':
    preprocess()
