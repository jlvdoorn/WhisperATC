from whisper.normalizers import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()

import re
import os

nato_alphabet_mapping       = {'A': 'alpha', 'B': 'bravo', 'C': 'charlie', 'D': 'delta', 'E': 'echo', 
                            'F': 'foxtrot', 'G': 'golf', 'H': 'hotel', 'I': 'india', 'J': 'juliett',
                            'K': 'kilo', 'L': 'lima', 'M': 'mike', 'N': 'november', 'O': 'oscar',
                            'P': 'papa', 'Q': 'quebec', 'R': 'romeo', 'S': 'sierra', 'T': 'tango',
                            'U': 'uniform', 'V': 'victor', 'W': 'whiskey', 'X': 'xray', 'Y': 'yankee', 'Z': 'zulu',
                         
                            '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
                            '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten', 
                            '0': 'zero', '00': 'hundred', '000': 'thousand',
                         
                            '.': 'decimal', ',': 'comma', '-': 'dash',}
nato_similarities           = {'alfa': 'alpha', 'oskar': 'oscar', 'ekko': 'echo', 'gulf': 'golf'}
terminology_mapping         = {'FL': 'flight level'}
text_similarities           = {'descent': 'descend'}

# Not needed for WER calculations
# airlines_icao_mapping       = {'lufthansa': 'lufthansa', 'speedbird': 'british airways'}
# airlines_synonym_mapping    = {'hansa': 'lufthansa'}

# Sometimes Whisper is intelligent enough to perceive 'eurowings seven alpha bravo' as 'EW7AB'
airlines_iata_codes         = {'BA': 'british airways', 'KL': 'klm', 'LH': 'lufthansa', 'EW': 'eurowings'}
airlines_icao_codes         = {'BAW': 'british airways', 'DLH': 'lufthansa', 'KLM': 'klm', 'EWG': 'eurowings'}

def aerospaceTransform(text):
    wrds = text.split()
    for word in wrds:
        if word in nato_alphabet_mapping:
            x = wrds.index(word)
            wrds[x] = nato_alphabet_mapping[word]
        if word.lower() in nato_similarities:
            x = wrds.index(word)
            wrds[x] = nato_similarities[word.lower()]
        if word in terminology_mapping:
            x = wrds.index(word)
            wrds[x] = terminology_mapping[word]
        if word.lower() in text_similarities:
            x = wrds.index(word)
            wrds[x] = text_similarities[word.lower()]
        if word.upper() in airlines_iata_codes:
            x = wrds.index(word)
            wrds[x] = airlines_iata_codes[word.upper()]            
        if word.upper() in airlines_icao_codes:
            x = wrds.index(word)
            wrds[x] = airlines_icao_codes[word.upper()]
    return ' '.join(wrds)

normalizer = EnglishTextNormalizer()

def removePunctuation(text):
    text = ''.join(
        ' ' if c in '!@#$%^&*~-+=_\|;:,.?' else c
        for c in text
    )
    return text

def separateNumbersAndText(text):
    text = re.split('(\d+)', text)
    text = ' '.join(text)
    return text

def separateCallSignLetters(text):
    wrds = text.split()
    prohibited_words = ['ILS', 'IFR', 'FL']
    for word in wrds:
        if word.isupper() and word not in prohibited_words:
            ltrs = [str(l) for l in word]
            ltrs = ' '.join(str(l) for l in ltrs)
            x = wrds.index(word)
            wrds[x] = ltrs
    
    return ' '.join(wrds)

def splitNumbersIntoDigits(text):
    wrds = text.split()
    for word in wrds:
        if word.isnumeric():
            dgts = [int(d) for d in word]
            dgts = ' '.join(str(d) for d in dgts)
            x = wrds.index(word)
            wrds[x] = dgts
        
    return ' '.join(wrds)

    # elements = re.split(' ', text)

def removeSpokenSeparators(text):
    wrds = text.split()
    for word in wrds:
        if word.lower() in ['decimal', 'comma', 'point']:
            x = wrds.index(word)
            wrds[x] = ''
        
    return ' '.join(wrds)

def splitGreetings(text):
    wrds = text.split()
    for word in wrds:
        if word.lower() in ['goodbye']:
            x = wrds.index(word)
            wrds[x] = 'good bye'
            
    return ' '.join(wrds)

def removeCharSet(text, c1, c2): # for removing all text within (and including) a character set (ex.: [TRANSCRIPT] )
    while c1 in text and c2 in text:
        x = text.find(c1)
        y = text.rfind(c2) # Should be the last entry of the closing element ) ] > 
        text = text[0:x] + text[y+1:]
    return text

def removeChar(text, c1): # for removing a single character (ex.: @ )
    while c1 in text:
        x = text.find(c1)
        text = text[0:x] + text[x+1:]
    return text

def removeNonAlphaNum(text): # for removing all non alphanumeric characters (ex.: ! @ # $ % ^ & * ) (AlphanNum.: A-Z, a-z, 0-9)
    for c in text:
        if c.isalnum() == False and c != ' ' :
            x = text.find(c)
            text = text[0:x] + text[x+1:]
    return text

def remove_multiples(string):
    for word in string.split():

        first = string.find(word)
        last = string.rfind(word)
        
        substring = string[first:last+len(word)]
        num = substring.count(word)
        if num > 5:
            new_substring = ' '.join([word for i in range(5)]).strip()
            string = string.replace(substring, new_substring)
            
    return string

def filterAndNormalize(text):   
    text = removeCharSet(text, '[', ']')
    text = removeCharSet(text, '<', '>')
    #text = removeCharSet(text, '(', ')')
    text = removeNonAlphaNum(text)
    text = separateNumbersAndText(text)
    text = aerospaceTransform(text)
    text = removeSpokenSeparators(text)
    # text = separateCallSignLetters(text)
    text = normalizer(text)
    text = normalizer(text)
    # Running twice because the normalizer will replace 'zero five' by '05' but also replaces '05' by '5' (removing leading zeros).
    text = splitNumbersIntoDigits(text)
    text = splitGreetings(text)
    #text = remove_multiples(text)
    text = text.lower()
    return text

def normalizeOnly(text):
    return normalizer(text)