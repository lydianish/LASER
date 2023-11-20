# This is a Python equivalent of the MOSES tooklit Perl script

import unicodedata

def remove_non_printing_chars(line):
    text = line.rstrip('\n')
    return ''.join(ch if unicodedata.category(ch)[0] != 'C' else ' ' for ch in text) + '\n'
