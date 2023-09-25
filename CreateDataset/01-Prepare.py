import glob
import os
import pandas as pd

from bs4 import BeautifulSoup

import datasets
from datasets import Audio
import re

## ATCO2-ASR
def prepare_atco2_asr():
    print('ATCO2-ASR')
    ###
    ## WAV : Audio
    ## XML : Transcription
    ## CNET: Probabilities (Ignored)
    ## INFO: Radar Data

    # 01. Read XML and extract transcription
    print('01. Read XML and extract transcription')

    xml_files = glob.glob('./ATCO2-ASR/DATA/*.xml')

    for xml_file in xml_files:
        with open(xml_file, 'r') as f:
            data = f.read()
            
        trs = ''
        bs_data = BeautifulSoup(data, 'xml')
        text_unique = bs_data.find_all('text') # All transcriptions are in <text> tags
        
        for tag in text_unique:
            trs += tag.text + ' '
            
        with open(xml_file.replace('.xml', '.trs'), 'w') as f:
            f.write(trs)
            
    # 02. Read INFO and extract radar data
    print('02. Read INFO and extract radar data')

    info_files = glob.glob('./ATCO2-ASR/DATA/*.info')

    for info_file in info_files:
        with open(info_file, 'r') as f:
            data = f.read().splitlines()
            
        info = ''
        for line in data:
            if line.startswith('airport'):
                airport = line.split(':')[1]
                airport_icao = airport[:airport.find('(')].strip()
                airport_name = airport[airport.find('(')+1:airport.find(')')].strip()
            if line.startswith('channel'):
                channel = line.split(':')[1]
            if line.startswith('waypoints nearby'):
                wpts_nearby = line.split(':')[1]
            if line.startswith('callsigns nearby'):
                line_num = data.index(line)
                break
            
        callsigns = data[line_num+1:]
        callsigns_abbr = []
        callsigns_full = []
        
        for callsign in callsigns:
            callsigns_abbr.append(callsign.split(':')[0].strip())
            if ':' in callsign:
                for item in callsign.split(':')[1].split(' '):
                    callsigns_full.append(item)
                    
        # remove duplicates from callsigns_abbr and callsigns_full
        callsigns_abbr = list(dict.fromkeys(callsigns_abbr))
        callsigns_full = list(dict.fromkeys(callsigns_full))
        
        callsigns_abbr = ' '.join(callsigns_abbr)
        callsigns_full = ' '.join(callsigns_full)

        radar_dict = airport_icao + '\n' + airport_name + '\n' + channel + '\n' + wpts_nearby + '\n' + callsigns_abbr + '\n' + callsigns_full
        with open(info_file.replace('.info', '.conv.info'), 'w') as f:
            f.write(radar_dict)
            
    # 03. Create Random Split
    print('03. Create Random Split')

    wav_files = glob.glob('./ATCO2-ASR/DATA/*.wav')

    os.system('mkdir ./ATCO2-ASR/DATA_SPLIT')
    os.system('mkdir ./ATCO2-ASR/DATA_SPLIT/train')
    os.system('mkdir ./ATCO2-ASR/DATA_SPLIT/validation')

    train_frac = 0.80
    validation_frac = 0.20

    train_wav_files = wav_files[:int(len(wav_files)*train_frac)]
    validation_wav_files = wav_files[int(len(wav_files)*train_frac):]

    for wav_file in train_wav_files:
        os.system('cp ' + wav_file + ' ./ATCO2-ASR/DATA_SPLIT/train/')
        os.system('cp ' + wav_file.replace('.wav', '.trs') + ' ./ATCO2-ASR/DATA_SPLIT/train/'+wav_file.split('/')[-1].replace('.wav', '.txt'))
        os.system('cp ' + wav_file.replace('.wav', '.conv.info') + ' ./ATCO2-ASR/DATA_SPLIT/train/'+wav_file.split('/')[-1].replace('.wav', '.info'))
        
    for wav_file in validation_wav_files:
        os.system('cp ' + wav_file + ' ./ATCO2-ASR/DATA_SPLIT/validation/')
        os.system('cp ' + wav_file.replace('.wav', '.trs') + ' ./ATCO2-ASR/DATA_SPLIT/validation/'+wav_file.split('/')[-1].replace('.wav', '.txt'))
        os.system('cp ' + wav_file.replace('.wav', '.conv.info') + ' ./ATCO2-ASR/DATA_SPLIT/validation/'+wav_file.split('/')[-1].replace('.wav', '.info'))
        
    # 04. Create Metadata CSV
    print('04. Create Metadata CSV')

    df = pd.DataFrame(columns=['file_name', 'text', 'info'])
    wav_files = glob.glob('./ATCO2-ASR/DATA_SPLIT/*/*.wav')
    wav_files.sort()

    for wav_file in wav_files:
        i = wav_files.index(wav_file)
        with open(wav_file.replace('.wav', '.txt'), 'r') as f:
            text = f.read()
        with open(wav_file.replace('.wav', '.info'), 'r') as f:
            info = f.read()
            
        df.loc[i] = ['/'.join(wav_file.split('/')[-2:]), text, info]
        
    df.to_csv('./ATCO2-ASR/DATA_SPLIT/metadata.csv', index=False)

    # 05. Create HuggingFace Dataset
    print('05. Create HuggingFace Dataset')

    # dataset = datasets.load_dataset("audiofolder", data_dir = "./ATCO2-ASR/DATA_SPLIT")

    # 06. Upload to HuggingFace Datasets Hub
    # print('06. Upload to HuggingFace Datasets Hub')
    # dataset.push_to_hub('ATCO2-ASR')

## ATCOSIM
def prepare_atcosim():
    print('ATCOSIM')
    ###
    ## WAVdata : Audio
    ## TXTdata : Transcription

    # 01. Filter TXT files - Purify Transcriptions
    print('01. Filter TXT files - Purify Transcriptions')

    txt_files = glob.glob('./ATCOSIM/TXTdata/*/*/*.txt')

    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            txt = f.read()
        txt = re.sub("[\(\<\[].*?[\>\)\]]", "", txt) # Remove all text within brackets ( ) [ ] < >
        # Remove files with empty transcriptions
        if txt.strip() == '':
            # os.system('rm ' + txt_file)
            # os.system('rm ' + txt_file.replace('TXTdata', 'WAVdata').replace('.txt', '.wav'))
            pass
        else:
            with open(txt_file.replace('.txt', '.filtered.txt'), 'w') as f:
                f.write(txt)
            
    # 02. Create Random Split 
    print('02. Create Random Split')

    wav_files = glob.glob('./ATCOSIM/WAVdata/*/*/*.wav')
    txt_files = glob.glob('./ATCOSIM/TXTdata/*/*/*.filtered.txt')
    
    os.system('mkdir ./ATCOSIM/DATA_SPLIT')
    os.system('mkdir ./ATCOSIM/DATA_SPLIT/train')
    os.system('mkdir ./ATCOSIM/DATA_SPLIT/validation')

    train_frac = 0.80
    validation_frac = 0.20

    train_txt_files = txt_files[:int(len(txt_files)*train_frac)]
    validation_txt_files = txt_files[int(len(txt_files)*train_frac):]

    for txt_file in train_txt_files:
        os.system('cp ' + txt_file + ' ./ATCOSIM/DATA_SPLIT/train/' + txt_file.split('/')[-1].replace('.filtered.txt', '.txt'))
        os.system('cp ' + txt_file.replace('TXTdata', 'WAVdata').replace('.filtered.txt', '.wav') + ' ./ATCOSIM/DATA_SPLIT/train/' + txt_file.split('/')[-1].replace('.filtered.txt', '.wav'))
        
    for txt_file in validation_txt_files:
        os.system('cp ' + txt_file + ' ./ATCOSIM/DATA_SPLIT/validation/' + txt_file.split('/')[-1].replace('.filtered.txt', '.txt'))
        os.system('cp ' + txt_file.replace('TXTdata', 'WAVdata').replace('.filtered.txt', '.wav') + ' ./ATCOSIM/DATA_SPLIT/validation/' + txt_file.split('/')[-1].replace('.filtered.txt', '.wav'))

    # 03. Create Metadata CSV
    print('03. Create Metadata CSV')

    df = pd.DataFrame(columns=['file_name', 'text'])

    wav_files = glob.glob('./ATCOSIM/DATA_SPLIT/*/*.wav')
    wav_files.sort()

    for wav_file in wav_files:
        i = wav_files.index(wav_file)
        with open(wav_file.replace('.wav', '.txt'), 'r') as f:
            text = f.read()
            
        df.loc[i] = ['/'.join(wav_file.split('/')[-2:]), text]
        
    df.to_csv('./ATCOSIM/DATA_SPLIT/metadata.csv', index=False)

    # 04. Create HuggingFace Dataset
    print('04. Create HuggingFace Dataset')

    # dataset = datasets.load_dataset("audiofolder", data_dir = "./ATCOSIM/DATA_SPLIT")

    # 05. Upload to HuggingFace Datasets Hub
    # print('05. Upload to HuggingFace Datasets Hub')
    # dataset.push_to_hub('ATCOSIM')

## ATCO2-ASR-ATCOSIM
def prepare_atco2_asr_atcosim():
    print('ATCO2-ASR-ATCOSIM')
    # 01. Merge ATCO2-ASR and ATCOSIM
    print('01. Merge ATCO2-ASR and ATCOSIM')

    train_wav = glob.glob('./*/DATA_SPLIT/train/*.wav')
    validation_wav = glob.glob('./*/DATA_SPLIT/validation/*.wav')

    os.system('mkdir ./ATCO2-ASR-ATCOSIM/')
    os.system('mkdir ./ATCO2-ASR-ATCOSIM/train')
    os.system('mkdir ./ATCO2-ASR-ATCOSIM/validation')


    for wav_file in train_wav:
        os.system('cp ' + wav_file + ' ./ATCO2-ASR-ATCOSIM/train/'+wav_file.split('/')[-1])
        os.system('cp ' + wav_file.replace('.wav', '.txt') + ' ./ATCO2-ASR-ATCOSIM/train/')
        os.system('cp ' + wav_file.replace('.wav', '.info') + ' ./ATCO2-ASR-ATCOSIM/train/')
        
    for wav_file in validation_wav:
        os.system('cp ' + wav_file + ' ./ATCO2-ASR-ATCOSIM/validation/'+wav_file.split('/')[-1])
        os.system('cp ' + wav_file.replace('.wav', '.txt') + ' ./ATCO2-ASR-ATCOSIM/validation/')
        os.system('cp ' + wav_file.replace('.wav', '.info') + ' ./ATCO2-ASR-ATCOSIM/validation/')
        
    # 02. Create Metadata CSV
    print('02. Create Metadata CSV')

    df = pd.DataFrame(columns=['file_name', 'text', 'info'])

    wav_files = glob.glob('./ATCO2-ASR-ATCOSIM/*/*.wav')
    wav_files.sort()

    for wav_file in wav_files:
        i = wav_files.index(wav_file)
        with open(wav_file.replace('.wav', '.txt'), 'r') as f:
            text = f.read()
        try:
            with open(wav_file.replace('.wav', '.info'), 'r') as f:
                info = f.read()
        except:
            info = ''
            
        df.loc[i] = ['/'.join(wav_file.split('/')[-2:]), text, info]
        
    df.to_csv('./ATCO2-ASR-ATCOSIM/metadata.csv', index=False)

    # 03. Create HuggingFace Dataset
    print('03. Create HuggingFace Dataset')

    # dataset = datasets.load_dataset("audiofolder", data_dir = "./ATCO2-ASR-ATCOSIM")

    # 04. Upload to HuggingFace Datasets Hub
    # print('04. Upload to HuggingFace Datasets Hub')
    # dataset.push_to_hub('ATCO2-ASR-ATCOSIM')
    
#prepare_atco2_asr()
prepare_atcosim()
prepare_atco2_asr_atcosim()