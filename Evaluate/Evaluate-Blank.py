# %% [markdown]
# # Infering Original and HF Whisper

# %%
dts = 'jlvdoorn/atco2-asr-atcosim'
mdl = 'openai/whisper-large-v3'
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
import whisper
model = whisper.load_model('-'.join(mdl.split('-')[1:]))

# %%
print('Starting inference...')
nato = "alpha,bravo,charlie,delta,echo,foxtrot,golf,hotel,india,juliett,kilo,lima,mike,november,oscar,papa,quebec,romeo,sierra,tango,uniform,victor,whiskey,xray,yankee,zulu"
terminology = "climb, climbing, descend, descending, passing, feet, knots, degrees, direct, maintain, identified, ILS, VFR, IFR, contact, frequency, turn, right, left, heading, altitude, flight, level, cleared, squawk, approach, runway, established, report, affirm, negative, wilco, roger, radio, radar"

for s in spl.split('+'):
    for i in range(len(dataset[s])):
        audio = dataset[s][i]['audio']['array']
        audio = whisper.pad_or_trim(audio)

        if wsp == 'large-v3':
            mel = whisper.log_mel_spectrogram(np.float32(audio), n_mels=128).to(model.device)
        else:
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
from Normalizer import filterAndNormalize

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
spl = 'train+validation'
wsp = '-'.join(mdl.split('-')[1:])

print('Dataset: ', dts)
print('Model  : ', mdl)
print('Split  : ', spl)
print('Whisper: ', wsp)

calcWER(df, spl)

# %%
# Split Validation
spl = 'validation'
wsp = '-'.join(mdl.split('-')[1:])

print('Dataset: ', dts)
print('Model  : ', mdl)
print('Split  : ', spl)
print('Whisper: ', wsp)

calcWER(df, spl)
