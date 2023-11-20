# This is a Python equivalent of the MOSES tooklit Perl script

import unicodedata

def rem_non_print_char(line):
    text = line.rstrip('\n')
    return ''.join(ch if unicodedata.category(ch)[0] != 'C' else ' ' for ch in text) + '\n'
