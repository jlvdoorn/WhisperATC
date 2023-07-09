import os
import re

import whisper
import glob
import pandas as pd

model = whisper.load_model('large-v2')

def initFiles(num_files=5): 
    # WAV files
    wav_atco2_asr = glob.glob('./WhisperModel/ATCO2-ASR/DATA/*.wav')
    wav_atco2_asr.sort()
    wav_atco2_asr = wav_atco2_asr[:num_files]

    wav_atcosim   = glob.glob('./WhisperModel/ATCOSIM/WAVdata/*/*/*.wav')
    wav_atcosim.sort()
    wav_atcosim = wav_atcosim[:num_files]

    wav_zcu       = glob.glob('./WhisperModel/ZCU_CZ_ATC/*.wav')
    wav_zcu.sort()
    wav_zcu = wav_zcu[:num_files]

    # Transcript files
    trs_atco2_asr = glob.glob('./WhisperModel/ATCO2-ASR/DATA/*.txt')
    trs_atco2_asr.sort()
    trs_atco2_asr = trs_atco2_asr[:num_files]

    trs_atcosim   = glob.glob('./WhisperModel/ATCOSIM/TXTdata/*/*/*.filtered.txt')
    trs_atcosim.sort()
    trs_atcosim = trs_atcosim[:num_files]

    trs_zcu       = glob.glob('./WhisperModel/ZCU_CZ_ATC/*.txt')
    trs_zcu.sort()
    trs_zcu = trs_zcu[:num_files]
    
    wav_files = {'ATCO2-ASR': wav_atco2_asr, 'ATCOSIM': wav_atcosim, 'ZCU': wav_zcu}
    trs_files = {'ATCO2-ASR': trs_atco2_asr, 'ATCOSIM': trs_atcosim, 'ZCU': trs_zcu}
    
    return wav_files, trs_files
    
def transcribe(prompts, wav_files, trs_files, idx):
    prompt_atco2_asr = prompts['ATCO2-ASR']
    prompt_atcosim = prompts['ATCOSIM']
    prompt_zcu = prompts['ZCU']
    
    wav_atco2_asr = wav_files['ATCO2-ASR']
    wav_atcosim = wav_files['ATCOSIM']
    wav_zcu = wav_files['ZCU']
  
    trs_atco2_asr = trs_files['ATCO2-ASR']
    trs_atcosim = trs_files['ATCOSIM']
    trs_zcu = trs_files['ZCU']  
    
    # Transcribing ATCO2 ASR
    result_prompt_atco2_asr = []
    result_clean_atco2_asr = []
    correct_atco2_asr = []

    print('ATCO2 - ASR')
    for i in range(len(wav_atco2_asr)):
        result = model.transcribe(wav_atco2_asr[i], initial_prompt=prompt_atco2_asr, language='en')
        result_prompt_atco2_asr.append(result['text'])

        result = model.transcribe(wav_atco2_asr[i], language='en')
        result_clean_atco2_asr.append(result['text'])
        
        with open(trs_atco2_asr[i], 'r') as f:
            correct = f.read()
            correct_atco2_asr.append(correct)
        print('Whisper:'+result['text'])
        print('Correct: '+correct)

    # Transcribing ATCOSIM
    result_prompt_atcosim = []
    result_clean_atcosim = []
    correct_atcosim = []

    print('ATCOSIM')
    for i in range(len(wav_atcosim)):
        result = model.transcribe(wav_atcosim[i], initial_prompt=prompt_atcosim, language='en')
        result_prompt_atcosim.append(result['text'])

        result = model.transcribe(wav_atcosim[i], language='en')
        result_clean_atcosim.append(result['text'])
        
        with open(trs_atcosim[i], 'r') as f:
            correct = f.read()
            correct_atcosim.append(correct)
        print('Whisper:'+result['text'])
        print('Correct: '+correct)

    # Transcribing ZCU
    result_prompt_zcu = []
    result_clean_zcu = []
    correct_zcu = []

    print('ZCU')
    for i in range(len(wav_zcu)):
        result = model.transcribe(wav_zcu[i], initial_prompt=prompt_zcu, language='en')
        result_prompt_zcu.append(result['text'])

        result = model.transcribe(wav_zcu[i], language='en')
        result_clean_zcu.append(result['text'])
        
        with open(trs_zcu[i], 'r') as f:
            correct = f.read()
            correct_zcu.append(correct)
        print('Whisper:'+result['text'])
        print('Correct: '+correct)

    # Framing the data
    data = pd.DataFrame(columns=['ATCO2-ASR-whisper-clean', 'ATCO2-ASR-whisper-prompt','ATCO2-ASR-correct', 'ATCOSIM-whisper-clean', 'ATCOSIM-whisper-prompt', 'ATCOSIM-correct', 'ZCU-whisper-clean', 'ZCU-whisper-prompt', 'ZCU-correct'])

    data['ATCO2-ASR-whisper-clean'] = result_clean_atco2_asr
    data['ATCO2-ASR-whisper-prompt'] = result_prompt_atco2_asr
    data['ATCO2-ASR-correct'] = correct_atco2_asr

    data['ATCOSIM-whisper-clean'] = result_clean_atcosim
    data['ATCOSIM-whisper-prompt'] = result_prompt_atcosim
    data['ATCOSIM-correct'] = correct_atcosim

    data['ZCU-whisper-clean'] = result_clean_zcu
    data['ZCU-whisper-prompt'] = result_prompt_zcu
    data['ZCU-correct'] = correct_zcu

    #data.T.to_excel('Whisper-Prompt-test.xlsx') # data is saved with file indexes as columns and data labels as row indexes for easier reading

    # try:
    #     data
    # except:
    #     import pandas as pd
    #     import numpy as np
    #     data = pd.read_excel('Whisper-Prompt-test.xlsx', header=None, index_col=0)
    #     data = data.T.drop(columns=np.nan)
    #     data.columns.name = None
    #     data = data.reset_index().drop(columns='index')

    data_atco2_asr = data[['ATCO2-ASR-whisper-clean', 'ATCO2-ASR-whisper-prompt', 'ATCO2-ASR-correct']]
    data_atco2_asr = data_atco2_asr.rename(columns={'ATCO2-ASR-whisper-clean': 'hyp-clean', 'ATCO2-ASR-whisper-prompt': 'hyp-prmpt', 'ATCO2-ASR-correct': 'ref'})

    data_atcosim   = data[['ATCOSIM-whisper-clean', 'ATCOSIM-whisper-prompt', 'ATCOSIM-correct']]
    data_atcosim   = data_atcosim.rename(columns={'ATCOSIM-whisper-clean': 'hyp-clean', 'ATCOSIM-whisper-prompt': 'hyp-prmpt', 'ATCOSIM-correct': 'ref'})

    data_zcu       = data[['ZCU-whisper-clean', 'ZCU-whisper-prompt', 'ZCU-correct']]
    data_zcu       = data_zcu.rename(columns={'ZCU-whisper-clean': 'hyp-clean', 'ZCU-whisper-prompt': 'hyp-prmpt', 'ZCU-correct': 'ref'})

    # Normalizing the data
    nato_alphabet_mapping       = {'A': 'alpha', 'B': 'beta', 'C': 'charlie', 'D': 'delta', 'E': 'echo', 
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

    from whisper.normalizers import EnglishTextNormalizer, BasicTextNormalizer
    normalizer = EnglishTextNormalizer() # EnglishTextNormalizer normalizer seems to result in better WER scores than BasicTextNormalizer

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
            y = text.find(c2) # Should be the last entry of the closing element ) ] > 
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

    data_atco2_asr.loc[:, 'hyp-clean-norm'] = data_atco2_asr.apply(lambda x: filterAndNormalize(x['hyp-clean']), axis=1)
    data_atco2_asr.loc[:, 'hyp-prmpt-norm'] = data_atco2_asr.apply(lambda x: filterAndNormalize(x['hyp-prmpt']), axis=1)
    data_atco2_asr.loc[:, 'ref-norm']       = data_atco2_asr.apply(lambda x: filterAndNormalize(x['ref']), axis=1)

    data_atcosim.loc[:, 'hyp-clean-norm']   = data_atcosim.apply(lambda x: filterAndNormalize(x['hyp-clean']), axis=1)
    data_atcosim.loc[:, 'hyp-prmpt-norm']   = data_atcosim.apply(lambda x: filterAndNormalize(x['hyp-prmpt']), axis=1)
    data_atcosim.loc[:, 'ref-norm']         = data_atcosim.apply(lambda x: filterAndNormalize(x['ref']), axis=1)

    data_zcu.loc[:, 'hyp-clean-norm']       = data_zcu.apply(lambda x: filterAndNormalize(x['hyp-clean']), axis=1)
    data_zcu.loc[:, 'hyp-prmpt-norm']       = data_zcu.apply(lambda x: filterAndNormalize(x['hyp-prmpt']), axis=1)
    data_zcu.loc[:, 'ref-norm']             = data_zcu.apply(lambda x: filterAndNormalize(x['ref']), axis=1)

    # Exporting Transcripts
    writer = pd.ExcelWriter('Transcripts-'+str(idx)+'.xlsx', engine='xlsxwriter')   
    data_atco2_asr.T.to_excel(excel_writer=writer, sheet_name='ATCO2-ASR')
    data_atcosim.T.to_excel(writer, sheet_name='ATCOSIM')
    data_zcu.T.to_excel(writer, sheet_name='ZCU')
    writer.save()

    # Calculating WER scores
    import jiwer

    # WER ( reference (correct) , hypothesis (whisper) )
    #wer_atco2_asr_clean  = jiwer.wer(list(data_atco2_asr['ref-norm']), list(data_atco2_asr['hyp-clean-norm']))
    wer_atco2_asr_prompt = jiwer.wer(list(data_atco2_asr['ref-norm']), list(data_atco2_asr['hyp-prmpt-norm']))

    #wer_atcosim_clean  = jiwer.wer(list(data_atcosim['ref-norm']), list(data_atcosim['hyp-clean-norm']))
    wer_atcosim_prompt = jiwer.wer(list(data_atcosim['ref-norm']), list(data_atcosim['hyp-prmpt-norm']))

    #wer_zcu_clean  = jiwer.wer(list(data_zcu['ref-norm']), list(data_zcu['hyp-clean-norm']))
    wer_zcu_prompt = jiwer.wer(list(data_zcu['ref-norm']), list(data_zcu['hyp-prmpt-norm']))

    import numpy as np
    print('Results for prompting:')
    print('ATCO2-ASR: ', prompt_atco2_asr)
    print('ATCOSIM:   ', prompt_atcosim)
    print('ZCU:       ', prompt_zcu)
    print()
    #print('ATCO2 ASR -- no prompt: {} %'.format(round(wer_atco2_asr_clean*100,2)))
    print('ATCO2 ASR --    prompt: {} %'.format(round(wer_atco2_asr_prompt*100,2)))
    print()
    #print('ATCOSIM   -- no prompt: {} %'.format(round(wer_atcosim_clean*100,2)))
    print('ATCOSIM   --    prompt: {} %'.format(round(wer_atcosim_prompt*100,2)))
    print()
    #print('ZCU       -- no prompt: {} %'.format(round(wer_zcu_clean*100,2)))
    print('ZCU       --    prompt: {} %'.format(round(wer_zcu_prompt*100,2)))
    print()

    #wer_avg_clean = np.average([wer_atco2_asr_clean, wer_atcosim_clean, wer_zcu_clean])
    wer_avg_prompt = np.average([wer_atco2_asr_prompt, wer_atcosim_prompt, wer_zcu_prompt])
    #print('Average   -- no prompt: {} %'.format(round(wer_avg_clean*100,2)))
    print('Average   --    prompt: {} %'.format(round(wer_avg_prompt*100,2)))
    
    wer = [wer_atco2_asr_prompt, wer_atcosim_prompt, wer_zcu_prompt, wer_avg_prompt]
    return wer


wav_files, trs_files = initFiles(5) 


# running takes around 35 minutes per prompt --> Use that in the .sh file to request the time.
all_prompts = ['Air Traffic',
               'Air Traffic Control communications with possible airlines Lufthansa, Eurowings, KLM, Aeroflot, Luxair']
wers = {}
for prompt in all_prompts:
    prompts = {'ATCO2-ASR': prompt, 'ATCOSIM': prompt, 'ZCU': prompt}
    idx = all_prompts.index(prompt)
    wer = transcribe(prompts, wav_files, trs_files, idx)
    wers[idx] = wer
    
df = pd.DataFrame(wers)
df.to_excel('WER-PROMPT-TESTING.xlsx')