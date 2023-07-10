# WhisperATC
Applying Large-Scale Weakly-Supervised Automatic Speech Recognition to Air Traffic Control

## Data Preparation
The CreateDataset module contains the necessary files to create and upload datasets to huggingface. A random split is made in the data where 80% will be used for training and 20% for validaiton. Currently the following datasets are available:

- ATCO2-ASR
- ATCOSIM
- ATCO2-ASR-ATCOSIM
- ZCU-CZ-ATC

However, the latter is not used due to the quality of the transcriptions. 

## Prompt Testing
The PromptTesting module contains the files used for the iterative experiments on prompting and normalization. 

## Evaluation
Two scripts are available for evaluating the models. One is for evaluating the blank models and the second is for evaulating the fine-tuned models. The fine-tuned models will be converted into the whisper format in order to be used for inference.

## Fine-Tuning
The fine-tuning scripts are created to form a modular way of fine-tuning the blank Whisper models on the created datasets. The models will automatically be uploaded into the huggingface format. The fine-tuning relies on the ```deepspeed``` package. Currently the following fine-tuned models are available:

- Whisper Large v2 - ATCO2
- Whisper Large v2 - ATCOSIM
- Whisper Large v2 - ATCO2 ATCOSIM

## Graphs
The Graphs module exists to create some graphs for for example a report or a presentation.

## HuggingFace🤗
All the datasets and models are available on the [HuggingFace🤗 Hub](https://huggingface.co/jlvdoorn).

## Environment
The repo contains a ```requirements.txt``` file. This can be used to create the needed conda environment with:
```
conda create --name whisper --file requirements.txt
```