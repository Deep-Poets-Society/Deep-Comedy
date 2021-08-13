import re
import pickle
import string
import warnings
import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

CESURA = " C "
SPACE = " S "
SYL = " Y "


def remove_useless_spaces(text):
    # remove multiple spaces
    processed_text = re.sub(r' +', f' ', text)
    # remove spaces at the beginning of each line
    processed_text = re.sub(r'^ ', '', processed_text)
    processed_text = re.sub(r'\n ', '\n', processed_text)
    
    return processed_text


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
    # remove multiple spaces and transform spaces in space tag
    processed_text = re.sub(r' +', f'{SPACE}', processed_text)

    # add syllable token in y.csv, remove syllabification in X.csv
    x_text = re.sub(r'\|', '', processed_text)
    y_text = re.sub(r'\|', f'{SYL}', processed_text)
    # remove spaces at the beginning of each line in y.csv
    y_text = re.sub(r'^ ', '', y_text)
    # generate y_cesura.csv
    y_cesura_text = cesura(x_text, y_text)
    # re-add syllable tag the beginning of the first syllable of each line
    y_cesura_text = re.sub(r'\n', f'\n{SYL}', y_cesura_text)
    # add an initial space which is lost in the process:
    y_cesura_text = ' ' + y_cesura_text

    # save files
    with open('res/X.csv', 'w+', encoding='utf-8') as file:
        file.writelines(x_text)
    with open('res/y.csv', 'w+', encoding='utf-8') as file:
        file.writelines(y_text)
    with open('res/y_cesura.csv', 'w+', encoding='utf-8') as file:
        file.writelines(y_cesura_text)

    text_no_tag = re.sub(rf'(\[START] )|( \[END])|({SYL})|({SPACE})|({CESURA})', ' ', y_text)
    text_no_tag = remove_useless_spaces(text_no_tag)
    create_vocabulary(text_no_tag)


def cesura(text, syl_text):
    dictionary = pickle.load(open('res/dantes_dictionary.pkl', 'rb'))
    # add entries not present
    dictionary.update({'aegypto': [((0, 3, -1, 0), 'ae|gyp|to', 1)]})
    # correct some inaccuracy
    dictionary['gorgoglian'] = [((0, 3, -1, 0), 'gor|go|glian', 1)]
    for word in ['te', 'beati', 'qui', 'neque', 'labïa', 'gloria']:
        dictionary[f'’{word}'] = dictionary[f'‘{word}']
    proclitic_worlds = ['il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'mi', 'ti', 'ci', 'vi', 'si', 'ne', 'e', 'o', 'ma',
                        'se', 'di', 'a', 'da', 'in', 'con', 'per', 'tra', 'fra', '‘', '‘l', 'gl‘',
                        'ch‘', 'm‘', 's‘', 'v‘', 'c‘', 't‘']
    # also 'che', 'non' and 'su' are usually considered proclitic, but in more than one verse in Dante it is considered
    # as stressed and determinant for cesura, so is omitted from the list,
    # like 'o tosco che | per la città del foco' (If. X, 22) 2-4t||8
    text = re.sub(r'(\[START] )|( \[END])', '', text)
    syl_text = re.sub(rf'(\[START]{SYL})|( \[END])', '', syl_text)
    lines = text.split('\n')
    syl_lines = syl_text.split('\n')
    lines_wo_cesura = 0
    for i, (line, syl_line) in enumerate(zip(lines, syl_lines)):
        tonic_accents = []
        counter = 0
        words = line.split(f'{SPACE}')
        for w in words:
            num_syl = dictionary[w][0][0][1]
            pos_accent = dictionary[w][0][0][2]
            tonic_accents += [False for _ in range(num_syl)]
            if w not in proclitic_worlds:
                tonic_accents[pos_accent - 1] = True
        syllables = syl_line.split(f'{SYL}')
        if '' in syllables:
            syllables.remove('')
        # Check if a sinalefe is present, then remove the right syllable(s)
        if len(syllables) != len(tonic_accents):
            for j, s in enumerate(syllables):
                for _ in re.findall(rf'{SPACE}.', s):
                    del tonic_accents[j]

        # hendecasyllable a maiore, cesura maschile
        if tonic_accents[5] and syllables[5].endswith(f'{SPACE}'):
            line_w_cesura = f'{SYL}'.join(syllables[0:6]) + f'{CESURA}' + f'{SYL}'.join(syllables[6:])
        # hendecasyllable a maiore, cesura femminile
        elif tonic_accents[5] and syllables[6].endswith(f'{SPACE}'):
            line_w_cesura = f'{SYL}'.join(syllables[0:7]) + f'{CESURA}' + f'{SYL}'.join(syllables[7:])
        # hendecasyllable a minore, cesura maschile
        elif tonic_accents[3] and syllables[3].endswith(f'{SPACE}'):
            line_w_cesura = f'{SYL}'.join(syllables[0:4]) + f'{CESURA}' + f'{SYL}'.join(syllables[4:])
        # hendecasyllable a minore, cesura femminile
        elif tonic_accents[3] and syllables[4].endswith(f'{SPACE}'):
            line_w_cesura = f'{SYL}'.join(syllables[0:5]) + f'{CESURA}' + f'{SYL}'.join(syllables[5:])
        # cesura lirica
        elif tonic_accents[2] and tonic_accents[5] and syllables[3].endswith(f'{SPACE}'):
            line_w_cesura = f'{SYL}'.join(syllables[0:4]) + f'{CESURA}' + f'{SYL}'.join(syllables[4:])
        # hendecasyllable a maiore, cesura after a proparoxytone
        elif tonic_accents[5] and syllables[7].endswith(f'{SPACE}'):
            line_w_cesura = f'{SYL}'.join(syllables[0:8]) + f'{CESURA}' + f'{SYL}'.join(syllables[8:])
        # hendecasyllable a minore, cesura after a proparoxytone
        elif tonic_accents[3] and syllables[5].endswith(f'{SPACE}'):
            line_w_cesura = f'{SYL}'.join(syllables[0:6]) + f'{CESURA}' + f'{SYL}'.join(syllables[6:])
        # hendecasyllable a maiore with cesura femminile between two words joint by sinalefe
        elif tonic_accents[5] and f'{SPACE}' in syllables[6]:
            line_w_cesura = f'{SYL}'.join(syllables[0:6]) + f'{SYL}' + \
                            re.sub(rf'{SPACE}(?=.)', rf'{SPACE}{CESURA}', syllables[6]) + \
                            f'{SYL}' + f'{SYL}'.join(syllables[7:])
        # hendecasyllable a minore with cesura femminile between two words joint by sinalefe
        elif tonic_accents[3] and f'{SPACE}' in syllables[4]:
            line_w_cesura = f'{SYL}'.join(syllables[0:4]) + f'{SYL}' + \
                            re.sub(rf'{SPACE}(?=.)', f'{SPACE}{CESURA}', syllables[4]) + \
                            f'{SYL}' + f'{SYL}'.join(syllables[5:])
        # hendecasyllable a maiore with cesura after a proparoxytone and between two words joint by sinalefe
        elif tonic_accents[5] and f'{SPACE}' in syllables[7]:
            line_w_cesura = f'{SYL}'.join(syllables[0:7]) + f'{SYL}' + \
                            re.sub(rf'{SPACE}(?=.)', f'{SPACE}{CESURA}', syllables[7]) + \
                            f'{SYL}' + f'{SYL}'.join(syllables[8:])
        # hendecasyllable a minore with cesura after a proparoxytone and between two words joint by sinalefe
        elif tonic_accents[3] and f'{SPACE}' in syllables[5]:
            line_w_cesura = f'{SYL}'.join(syllables[0:5]) + f'{SYL}' + \
                            re.sub(rf'{SPACE}(?=.)', f'{SPACE}{CESURA}', syllables[5]) + \
                            f'{SYL}' + f'{SYL}'.join(syllables[6:])
        # There are 4 hendecasyllable a maiore where the cesura split a compound word which ends with -mente (3
        # cases) or -zial (1 case)
        elif 'mente' in syllables[7] + syllables[8]:
            line_w_cesura = f'{SYL}'.join(syllables[0:7]) + f'{CESURA}' + f'{SYL}'.join(syllables[7:])
        elif 'zial ' in syllables[6] + syllables[7]:
            line_w_cesura = f'{SYL}'.join(syllables[0:6]) + f'{CESURA}' + f'{SYL}'.join(syllables[6:])
        # Remaining 7 verses are non-canonical hendecasyllables so they do not have a cesura
        else:
            lines_wo_cesura += 1
            line_w_cesura = syl_line
        # remove eventual multiple spaces
        line_w_cesura = remove_useless_spaces(line_w_cesura)
        syl_lines[i] = line_w_cesura

    if lines_wo_cesura != 7:
        warnings.warn(
            f'Warning! There are {lines_wo_cesura} lines without cesura, they should be only 7',
            UserWarning)
    return '\n'.join(syl_lines)


def create_vocabulary(text):
    y = tf.data.Dataset.from_tensor_slices(text.split('\n'))
    tokenizer_params = dict(lower_case=False)
    reserved_tokens = ['[START]', '[END]', f'{SYL}'.strip(), f'{CESURA}'.strip(), f'{SPACE}'.strip()]
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
