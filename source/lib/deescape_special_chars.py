# This is a Python equivalent of the MOSES tooklit Perl script

import html

def deescape_special_chars(line):
    text = html.unescape(line.rstrip('\n'))
    text = text.replace('&bar;', '|')  # factor separator (legacy)
    text = text.replace('&#124;', '|')  # factor separator
    text = text.replace('&lt;', '<')  # xml
    text = text.replace('&gt;', '>')  # xml
    text = text.replace('&bra;', '[')  # syntax non-terminal (legacy)
    text = text.replace('&ket;', ']')  # syntax non-terminal (legacy)
    text = text.replace('&quot;', '"')  # xml
    text = text.replace('&apos;', "'")  # xml
    text = text.replace('&#91;', '[')  # syntax non-terminal
    text = text.replace('&#93;', ']')  # syntax non-terminal
    text = text.replace('&amp;', '&')  # escape escape
    return text
