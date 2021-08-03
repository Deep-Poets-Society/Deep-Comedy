import re


def preprocess():
    with open('res/divina_commedia_raw.txt', 'r+', encoding='utf-8') as file:
        raw_text = ''.join(file.readlines())
    # remove capital letters
    processed_text = raw_text.lower()
    # substitute ’ with '
    processed_text = re.sub('’', '\'', processed_text)
    # remove punctuation, but '
    processed_text = re.sub(r'[,.;:\-—?!"’()\[\]«»“‟”]', '', processed_text)
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

    # save files
    with open('res/X.csv', 'w+', encoding='utf-8') as file:
        file.writelines(x_text)
    with open('res/y.csv', 'w+', encoding='utf-8') as file:
        file.writelines(y_text)


if __name__ == '__main__':
    preprocess()
