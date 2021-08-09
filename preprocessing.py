import re
import pickle


def preprocess():
    with open('res/divina_commedia_raw.txt', 'r+', encoding='utf-8') as file:
        raw_text = ''.join(file.readlines())
    # remove capital letters
    processed_text = raw_text.lower()
    # remove punctuation, but ’
    processed_text = re.sub(r'[,.;:\-—?!"()\[\]«»“‟”]', '', processed_text)
    # remove cantos' headings
    processed_text = re.sub(r'.* • canto .*', '', processed_text)
    # remove lines' numbers
    processed_text = re.sub(r'\n *\d* ', '\n', processed_text)
    # remove multiple blank lines
    processed_text = re.sub(r'\n+', '\n', processed_text)
    # remove initial and last blank lines
    processed_text = re.sub(r'(^\n)|(\n$)', '', processed_text)
    # remove multiple spaces and transform spaces in <s>
    processed_text = re.sub(r' +', '<s>', processed_text)
    # add <start> and <end> tokens
    processed_text = re.sub(r'\n', r'<end>\n<start>', processed_text)
    processed_text = re.sub('^', '<start>', processed_text)
    processed_text = re.sub('$', '<end>', processed_text)

    # add <syl> token in y.csv, remove syllabification in X.csv
    x_text = re.sub(r'\|', '', processed_text)
    y_text = re.sub(r'\|', r'<syl>', processed_text)
    # generate y_cesura.csv
    y_cesura_text = cesura(x_text, y_text)

    # save files
    with open('res/X.csv', 'w+', encoding='utf-8') as file:
        file.writelines(x_text)
    with open('res/y.csv', 'w+', encoding='utf-8') as file:
        file.writelines(y_text)
    with open('res/y_cesura.csv', 'w+', encoding='utf-8') as file:
        file.writelines(y_cesura_text)


def cesura(text, syl_text):
    dictionary = pickle.load(open('res/dantes_dictionary.pkl', 'rb'))
    # add entries not present
    dictionary.update({'aegypto': [((0, 3, -1, 0), 'ae|gyp|to', 1)]})
    text = re.sub(r'(<start>)|(<end>)', '', text)
    syl_text = re.sub(r'(<start><syl>)|(<end>)', '', syl_text)
    lines = text.split('\n')
    syl_lines = syl_text.split('\n')
    for i, (line, syl_line) in enumerate(zip(lines, syl_lines)):
        tonic_accents = []
        counter = 0
        words = line.split('<s>')
        for w in words:
            num_syl = dictionary[w][0][0][1]
            pos_accent = dictionary[w][0][0][2]
            tonic_accents += [False for _ in range(num_syl)]
            tonic_accents[pos_accent - 1] = True
        syllables = syl_line.split('<syl>')
        if len(syllables) != len(tonic_accents):
            for j, s in enumerate(syllables):
                for _ in re.findall(r'<s>.', s):
                    del tonic_accents[j]
        # cesura lirica
        if tonic_accents[2] and tonic_accents[5] and syllables[3].endswith('<s>'):
            line_w_cesura = '<syl>'.join(syllables[0:4]) + '<syl><c>' + '<syl>'.join(syllables[4:])
        # hendecasyllable a maiore, cesura maschile
        elif tonic_accents[5] and syllables[5].endswith('<s>'):
            line_w_cesura = '<syl>'.join(syllables[0:6]) + '<syl><c>' + '<syl>'.join(syllables[6:])
        # hendecasyllable a minore, cesura maschile
        elif tonic_accents[3] and syllables[3].endswith('<s>'):
            line_w_cesura = '<syl>'.join(syllables[0:4]) + '<syl><c>' + '<syl>'.join(syllables[4:])
        # hendecasyllable a maiore, cesura femminile
        elif tonic_accents[5] and syllables[6].endswith('<s>'):
            line_w_cesura = '<syl>'.join(syllables[0:7]) + '<syl><c>' + '<syl>'.join(syllables[7:])
        # hendecasyllable a minore, cesura femminile
        elif tonic_accents[3] and syllables[4].endswith('<s>'):
            line_w_cesura = '<syl>'.join(syllables[0:5]) + '<syl><c>' + '<syl>'.join(syllables[5:])
        # hendecasyllable a maiore, cesura after a proparoxytone
        elif tonic_accents[5] and syllables[7].endswith('<s>'):
            line_w_cesura = '<syl>'.join(syllables[0:8]) + '<syl><c>' + '<syl>'.join(syllables[8:])
        # hendecasyllable a minore, cesura after a proparoxytone
        elif tonic_accents[3] and syllables[5].endswith('<s>'):
            line_w_cesura = '<syl>'.join(syllables[0:6]) + '<syl><c>' + '<syl>'.join(syllables[6:])
        # hendecasyllable a maiore with cesura femminile between two words joint by sinalefe
        elif tonic_accents[5] and '<s>' in syllables[6]:
            line_w_cesura = '<syl>'.join(syllables[0:6]) + '<syl>' + \
                            re.sub(r'<s>(?=.)', r'<s><c>', syllables[6]) + \
                            '<syl>' + '<syl>'.join(syllables[7:])
        # hendecasyllable a minore with cesura femminile between two words joint by sinalefe
        elif tonic_accents[3] and '<s>' in syllables[4]:
            line_w_cesura = '<syl>'.join(syllables[0:4]) + '<syl>' + \
                            re.sub(r'<s>(?=.)', r'<s><c>', syllables[4]) + \
                            '<syl>' + '<syl>'.join(syllables[5:])
        # hendecasyllable a maiore with cesura after a proparoxytone and between two words joint by sinalefe
        elif tonic_accents[5] and '<s>' in syllables[7]:
            line_w_cesura = '<syl>'.join(syllables[0:7]) + '<syl>' + \
                            re.sub(r'<s>(?=.)', r'<s><c>', syllables[7]) + \
                            '<syl>' + '<syl>'.join(syllables[8:])
        # hendecasyllable a minore with cesura after a proparoxytone and between two words joint by sinalefe
        elif tonic_accents[3] and '<s>' in syllables[5]:
            line_w_cesura = '<syl>'.join(syllables[0:5]) + '<syl>' + \
                            re.sub(r'<s>(?=.)', r'<s><c>', syllables[5]) + \
                            '<syl>' + '<syl>'.join(syllables[6:])
        # There are 4 hendecasyllable a maiore where the cesura split a compound word which ends with -mente (3
        # cases) or -zial (1 case)
        elif 'mente<s>' == syllables[7] + syllables[8]:
            line_w_cesura = '<syl>'.join(syllables[0:7]) + '<syl><c>' + '<syl>'.join(syllables[7:])
        elif 'zial<s>' == syllables[6] +  syllables[7]:
            line_w_cesura = '<syl'.join(syllables[0:6]) + '<syl><c>' + '<syl>'.join(syllables[6:])
        # Remaining 7 verses are non-canonical hendecasyllables so they do not have a cesura
        else:
            line_w_cesura = re.sub(r'(<start><syl>)|(<end>)', '', syl_line)
        line_w_cesura = '<start><syl>' + line_w_cesura + '<end>'
        syl_lines[i] = line_w_cesura
    return '\n'.join(syl_lines)


if __name__ == '__main__':
    preprocess()
