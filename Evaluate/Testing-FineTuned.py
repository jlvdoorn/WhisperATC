# %% [markdown]
# # Infering Original and HF Whisper

# %%
dts = 'jlvdoorn/atco2-asr'
mdl = 'jlvdoorn/whisper-large-v2-atcosim'
spl = 'train+validation'
wsp = '-'.join(mdl.split('-')[1:])

print('Dataset: ', dts)
print('Model  : ', mdl)
print('Split  : ', spl)
print('Whisper: ', wsp)

# %%
from datasets import load_dataset, Audio
dataset = load_dataset(dts)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset

# %%
import numpy as np
import pandas as pd
from datetime import datetime

# %%
df = pd.DataFrame(columns=['split', 'hyp-prmpt', 'hyp-clean', 'ref'])

# %% [markdown]
# # Infering Original Whisper with HF Dataset

# %%
import re
import whisper
import torch

def hf_to_whisper_states(text):
    text = re.sub('.layers.', '.blocks.', text)
    text = re.sub('.self_attn.', '.attn.', text)
    text = re.sub('.q_proj.', '.query.', text)
    text = re.sub('.k_proj.', '.key.', text)
    text = re.sub('.v_proj.', '.value.', text)
    text = re.sub('.out_proj.', '.out.', text)
    text = re.sub('.fc1.', '.mlp.0.', text)
    text = re.sub('.fc2.', '.mlp.2.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.encoder_attn.', '.cross_attn.', text)
    text = re.sub('.cross_attn.ln.', '.cross_attn_ln.', text)
    text = re.sub('.embed_positions.weight', '.positional_embedding', text)
    text = re.sub('.embed_tokens.', '.token_embedding.', text)
    text = re.sub('model.', '', text)
    text = re.sub('attn.layer_norm.', 'attn_ln.', text)
    text = re.sub('.final_layer_norm.', '.mlp_ln.', text)
    text = re.sub('encoder.layer_norm.', 'encoder.ln_post.', text)
    text = re.sub('decoder.layer_norm.', 'decoder.ln.', text)
    return text

hf_state_dict = torch.load('/scratch/junzisun/'+mdl.split('/')[-1]+'/pytorch_model.bin')    # pytorch_model.bin file

# Rename layers
for key in list(hf_state_dict.keys())[:]:
    new_key = hf_to_whisper_states(key)
    hf_state_dict[new_key] = hf_state_dict.pop(key)

# Init Whisper Model and replace model weights
model = whisper.load_model('large-v2')
model.load_state_dict(hf_state_dict)

# %%
nato = "alpha,bravo,charlie,delta,echo,foxtrot,golf,hotel,india,juliett,kilo,lima,mike,november,oscar,papa,quebec,romeo,sierra,tango,uniform,victor,whiskey,xray,yankee,zulu"
terminology = "climb, climbing, descend, descending, passing, feet, knots, degrees, direct, maintain, identified, ILS, VFR, IFR, contact, frequency, turn, right, left, heading, altitude, flight, level, cleared, squawk, approach, runway, established, report, affirm, negative, wilco, roger, radio, radar"

for s in spl.split('+'):
    print(' ')
    for i in range(len(dataset[s])):
        audio = dataset[s][i]['audio']['array']
        audio = whisper.pad_or_trim(audio)
        
        mel = whisper.log_mel_spectrogram(np.float32(audio)).to(model.device)
        
        try:
            prompt = 'Air Traffic Control Communications ' + dataset[s][i]['info'].replace('\n', ' ') + ' ' + nato.replace(',',' ') + ' ' + terminology.replace(',',' ')

        except:
            inf = ''
            prompt = 'Air Traffic Control Communications ' + nato.replace(',',' ') + ' ' + terminology.replace(',',' ')
            
        options = whisper.DecodingOptions(language='en', prompt=prompt, fp16=False)
        res_prmpt = whisper.decode(model, mel, options=options)
        options = whisper.DecodingOptions(language='en', fp16=False)
        res_clean = whisper.decode(model, mel, options=options)
        
        df.loc[len(df.index)] = [s, res_prmpt.text, res_clean.text, dataset[s][i]['text']]
        
        print(s, str(int(i/len(dataset[s])*100))+'%', end='\r')
df.to_excel(dts.split('/')[-1]+'-'+spl+'-'+mdl.split('/')[-1]+'-'+datetime.today().strftime('%Y-%m-%d--%H:%M:%S')+'.xlsx')        

# %%
df

# %% [markdown]
# # Normalization

# %%
from whisper.normalizers import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()

# %%
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
df['ref-norm'] = df.apply(lambda x: filterAndNormalize(x['ref']), axis=1)
df['hyp-clean-norm'] = df.apply(lambda x: filterAndNormalize(x['hyp-clean']), axis=1)
df['hyp-prmpt-norm'] = df.apply(lambda x: filterAndNormalize(x['hyp-prmpt']), axis=1)

# %%
df.head()

# %% [markdown]
# # WER Calculation

# %%
import jiwer

# %%
def calcWER(df, spl):
    dff = df.loc[df['split'].isin(spl.split('+'))]
    wer_cln = jiwer.wer(list(dff['ref']), list(dff['hyp-clean']))
    wer_prm = jiwer.wer(list(dff['ref']), list(dff['hyp-prmpt']))
    wer_cln_nrm = jiwer.wer(list(dff['ref-norm']), list(dff['hyp-clean-norm']))
    wer_prm_nrm = jiwer.wer(list(dff['ref-norm']), list(dff['hyp-prmpt-norm']))

    print('clean        : {} %'.format(round(wer_cln*100,4)))
    print('prmpt        : {} %'.format(round(wer_prm*100,4)))
    print('clean-norm   : {} %'.format(round(wer_cln_nrm*100,4)))
    print('prmpt-norm   : {} %'.format(round(wer_prm_nrm*100,4)))

# %%
# Split Train+Validation
dts = 'jlvdoorn/atco2-asr'
mdl = 'jlvdoorn/whisper-large-v2-atcosim'
spl = 'train+validation'
wsp = '-'.join(mdl.split('-')[1:])

print('Dataset: ', dts)
print('Model  : ', mdl)
print('Split  : ', spl)
print('Whisper: ', wsp)

calcWER(df, spl)

# %%
# Split Validation
dts = 'jlvdoorn/atco2-asr'
mdl = 'jlvdoorn/whisper-large-v2-atcosim'
spl = 'validation'
wsp = '-'.join(mdl.split('-')[1:])

print('Dataset: ', dts)
print('Model  : ', mdl)
print('Split  : ', spl)
print('Whisper: ', wsp)

calcWER(df, spl)