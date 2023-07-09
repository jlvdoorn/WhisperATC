# %% [markdown]
# # Whisper ATCO2 Testing

# %%
import glob
import whisper
from whisper.normalizers import EnglishTextNormalizer
import re
import pandas as pd
import jiwer
import numpy as np

# %%
wav_files  = glob.glob('../../WhisperModel/ATCO2-ASR/DATA/*.wav')
wav_files.sort()
print(wav_files)
info_files = glob.glob('../../WhisperModel/ATCO2-ASR/DATA/*.conv.info')
info_files.sort()
txt_files  = glob.glob('../../WhisperModel/ATCO2-ASR/DATA/*.txt')
txt_files.sort()

# %%
def convertInfoFile(file):
    airport = ''
    airport_full = ''
    channel = ''
    wpts = ''
    csns = ''
    csns_full = ''
    
    with open(file, 'r') as f:
        info_full = f.read()
        csns = ''
        csns_full = ''
        for line in info_full.splitlines():
            if 'airport' in line:
                airport = line[9:13]
                x = line.find('(')
                y = line.find(')')
                airport_full = line[x+1:y]
            if 'channel' in line:
                channel = line[9:]
            if 'waypoints nearby' in line:
                wpts = line[18:]
            if 'callsigns nearby' in line:
                x = info_full.splitlines().index(line) + 1
                for line in info_full.splitlines()[x:]:
                    csn = line.split(' ')[0]
                    csns = csns + ' ' + csn
                    
                    csn_full = ' '.join(line.split(' ')[2:])
                    csns_full = csns_full + ' ' + csn_full
                csns = csns[1:]
                csns_full = csns_full[1:]
                
                csns_full = list(dict.fromkeys(csns_full.split(' ')))

    nato_alphabet =['alpha', 'bravo', 'charlie', 'delta', 'echo', 
                    'foxtrot', 'golf', 'hotel', 'india', 'juliett',
                    'kilo', 'lima', 'mike', 'november', 'oscar',
                    'papa', 'quebec', 'romeo', 'sierra', 'tango',
                    'uniform', 'victor', 'whiskey', 'xray', 'x-ray', 'yankee', 'zulu',
                         
                    'one', 'two', 'three', 'four', 'five',
                    'six', 'seven', 'eight', 'nine', 'ten', 
                    'zero', 'hundred', 'thousand']
    
    csns_full = [x for x in csns_full if x.lower() not in nato_alphabet and len(x)!=0]
    
    # print(airport)
    # print(channel)
    # print(wpts)
    # print(csns)
    # print(csns_full)
    # print(info_full)
    # return airport, channel, wpts, csns, csns_full

    file = file[:-4]+'conv.info'
    with open(file, 'x') as f:
        f.write(airport)
        f.write('\n')
        f.write(airport_full)
        f.write('\n')
        f.write(channel)
        f.write('\n')
        f.write(wpts)
        f.write('\n')
        f.write(csns)
        f.write('\n')
        f.write(' '.join(csns_full))

# %%
model = whisper.load_model('large-v2')

# %%
def readInfoFile(file):
    with open(file, 'r') as f:
        airport = " "
        airport_full = " "
        channel = " "
        wpts = " "
        csns = " "
        csns_full = " "
        lines = f.read().splitlines()
        try:
            airport = lines[0]
        except:
            pass
        try:
            airport_full = lines[1]
        except:
            pass
        try:
            channel = lines[2]
        except:
            pass
        try:
            wpts = lines[3]
        except:
            pass
        try:
            csns = lines[4]
        except:
            pass
        try:
            csns_full = lines[5]
        except:
            pass
        
    return airport, airport_full, channel, wpts, csns, csns_full

# %%
hyp_clean = []
hyp_prmpt = []
ref = []

for file in wav_files:
    prompt_general = "Air Traffic Control communications"
    airport, airport_full, channel, wpts, csns, csns_full = readInfoFile(file[:-3]+'conv.info')
    nato = "alpha,bravo,charlie,delta,echo,foxtrot,golf,hotel,india,juliett,kilo,lima,mike,november,oscar,papa,quebec,romeo,sierra,tango,uniform,victor,whiskey,xray,yankee,zulu"
    terminology = "climb, climbing, descend, descending, passing, feet, knots, degrees, direct, maintain, identified, ILS, VFR, IFR, contact, frequency, turn, right, left, heading, altitude, flight, level, cleared, squawk, approach, runway, established, report, affirm, negative, wilco, roger, radio, radar"
    prompt = prompt_general + " " + airport + " " + airport_full + " " + channel + " " + wpts + " " + csns + " " + csns_full + " " + nato + " " + terminology
    
    res_clean = model.transcribe(file, language='en')
    hyp_clean.append(res_clean)
    res_prmpt = model.transcribe(file, language='en', initial_prompt = prompt)
    hyp_prmpt.append(res_prmpt)
    
    with open(file[:-3]+'txt') as f:
        correct = f.read()
        ref.append(correct)

# %%
df = pd.DataFrame([hyp_clean, hyp_prmpt, ref]).T # columns: hyp-clean, hyp-prmpt, ref
df = df.rename(columns={0: 'hyp-clean', 1: 'hyp-prmpt', 2: 'ref'})

# %%
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

# %%
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

# %%
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
    
    text = text.lower()
    return text

def normalizeOnly(text):
    return normalizer(text)

# %%
df.loc[:, 'hyp-clean-norm'] = df.apply(lambda x: filterAndNormalize(x['hyp-clean']), axis=1)
df.loc[:, 'hyp-prmpt-norm'] = df.apply(lambda x: filterAndNormalize(x['hyp-prmpt']), axis=1)

writer = pd.ExcelWriter('Transcripts-ATCO2-ASR.xlsx', engine='xlsxwriter')   
df.T.to_excel(excel_writer=writer, sheet_name='ATCO2-ASR')
writer.save()

# %%
wer_clean = jiwer.wer(list(df['ref-norm']), list(df['hyp-clean-norm']))
wer_prmpt = jiwer.wer(list(df['ref-norm']), list(df['hyp-prmpt-norm']))
print('ATCO2 ASR -- no prompt: {} %'.format(round(wer_clean*100,2)))
print('ATCO2 ASR --    prompt: {} %'.format(round(wer_prmpt*100,2)))


