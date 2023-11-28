# %% [markdown]
# # Infering Original and HF Whisper

# %%
dts = 'jlvdoorn/atco2-asr'
mdl = 'jlvdoorn/whisper-large-v3-atco2-asr'
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
import os

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

if not os.path.exists(mdl.split('/')[-1]):
    os.system('git clone git@hf.co:'+mdl)
else:
    os.system('cd '+mdl.split('/')[-1]+' && git pull')
    os.system('cd '+mdl.split('/')[-1]+' && git lfs pull')
hf_state_dict = torch.load('./'+mdl.split('/')[-1]+'/pytorch_model.bin')    # pytorch_model.bin file

# Rename layers
for key in list(hf_state_dict.keys())[:]:
    new_key = hf_to_whisper_states(key)
    hf_state_dict[new_key] = hf_state_dict.pop(key)

# Init Whisper Model and replace model weights
model = whisper.load_model('large-v2')
model.load_state_dict(hf_state_dict)

# %%
print('Starting inference...')
nato = "alpha,bravo,charlie,delta,echo,foxtrot,golf,hotel,india,juliett,kilo,lima,mike,november,oscar,papa,quebec,romeo,sierra,tango,uniform,victor,whiskey,xray,yankee,zulu"
terminology = "climb, climbing, descend, descending, passing, feet, knots, degrees, direct, maintain, identified, ILS, VFR, IFR, contact, frequency, turn, right, left, heading, altitude, flight, level, cleared, squawk, approach, runway, established, report, affirm, negative, wilco, roger, radio, radar"

for s in spl.split('+'):
    print(' ')
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
