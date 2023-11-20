# This is a Python equivalent of the MOSES tooklit Perl script

import re

def normalize_punctuation(line, language="en", penn=0):
    text = line.replace('\r', '')
    
    # remove extra spaces
    text = re.sub(r'\(', ' (', text)
    text = re.sub(r'\)', ') ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\) ([\.\!\:\?\;\,])', r')\1', text)
    text = re.sub(r'\( ', '(', text)
    text = re.sub(r' \)', ')', text)
    text = re.sub(r'(\d) %', r'\1%', text)
    text = re.sub(r' :', ':', text)
    text = re.sub(r' ;', ';', text)

    # normalize unicode punctuation
    if penn == 0:
        text = text.replace('`', "'")
        text = text.replace("''", ' " ')

    text = text.replace('„', '"')
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    text = text.replace('–', '-')
    text = re.sub(r'—', ' - ', text)
    text = re.sub(r' +', ' ', text)
    text = text.replace('´', "'")
    text = re.sub(r'([a-z])‘([a-z])', r"\1'\2", text, flags=re.IGNORECASE)
    text = re.sub(r'([a-z])’([a-z])', r"\1'\2", text, flags=re.IGNORECASE)
    text = text.replace('‘', '"')
    text = text.replace('‚', '"')
    text = text.replace('’', '"')
    text = text.replace("''", '"')
    text = text.replace('´´', '"')
    text = text.replace('…', '...')
    
    # French quotes
    text = text.replace(' « ', ' "')
    text = text.replace('« ', '"')
    text = text.replace('«', '"')
    text = text.replace(' » ', '" ')
    text = text.replace(' »', '"')
    text = text.replace('»', '"')

    # handle pseudo-spaces
    text = text.replace(' %', '%')
    text = text.replace('nº ', 'nº ')
    text = text.replace(' :', ':')
    text = text.replace(' ºC', ' ºC')
    text = text.replace(' cm', ' cm')
    text = text.replace(' ?', '?')
    text = text.replace(' !', '!')
    text = text.replace(' ;', ';')
    text = text.replace(', ', ', ')
    text = re.sub(r' +', ' ', text)

    # English "quotation," followed by comma, style
    if language == "en":
        text = re.sub(r'\"([,\.]+)', r'\1"', text)
    # German/Spanish/French "quotation", followed by comma, style
    elif language in ["de", "es", "cz", "cs", "fr"]:
        text = re.sub(r',\"', '",', text)
        text = re.sub(r'(\.+)"(\s*[^<])', r'"\1\2', text)

    if language in ["de", "es", "cz", "cs", "fr"]:
        text = re.sub(r'(\d) (\d)', r'\1,\2', text)
    else:
        text = re.sub(r'(\d) (\d)', r'\1.\2', text)

    return text
