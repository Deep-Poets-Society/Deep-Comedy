import re
import string
from preprocessing import remove_useless_spaces, CESURA, SPACE, SYL

TRIPLET = "T"
NEW_LINE = "N"


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

    # add triplet token
    processed_text = re.sub(r'\n\n', f'{TRIPLET}\n', processed_text)
    # remove multiple blank lines
    processed_text = re.sub(r'\n+', '\n', processed_text)
    # remove initial and last blank lines
    processed_text = re.sub(r'(^\n)|(\n$)', '', processed_text)
    # remove multiple spaces and transform spaces in space tag
    processed_text = re.sub(r' +', f'{SPACE}', processed_text)


    processed_text = re.sub(r'\|', f'{SYL}', processed_text)


    # remove space tag if after triplet tag
    processed_text = re.sub(rf'{TRIPLET}{SPACE}', f'{TRIPLET}', processed_text)
    # remove syllabification
    processed_text = re.sub(r'\|', '', processed_text)

    # remove empty lines with triplet tag
    processed_text = re.sub(rf'\n{TRIPLET}\n{TRIPLET}\n', '\n', processed_text)
    # remove triplet tag at the beginning
    processed_text = re.sub(rf'^{TRIPLET}\n', '', processed_text)
    # add a space before triplet tag:
    processed_text = re.sub(f'{TRIPLET}', f' {TRIPLET}', processed_text)
    # add new line token
    processed_text = re.sub(r'\n', f' {NEW_LINE}\n', processed_text)
    processed_text = re.sub(rf'{TRIPLET} {NEW_LINE}', f'{TRIPLET}', processed_text)
    # save file
    with open('res/X_gen.csv', 'w+', encoding='utf-8') as file:
        file.writelines(processed_text)


if __name__ == '__main__':
    preprocess()