# Create Datasets

This folder is meant for the creation of the datasets of:

- ATCO2
- ATCOSIM
- ATCO2-ATCOSIM

## Requirements

```python
pandas
beautifulsoup4
datasets
```

The code is created for use on UNIX machines.

## Usage

1. The first step is to run ```00-Download.sh```. That script downloads the corpusses and extract the needed files.
2. Run ```01-Prepare.py``` to pre-proces the datasets and upload them to the [HuggingFace🤗 Hub](https://hf.co/).

## HuggingFace

The pre-processed versions of the datasets are available publicly on the HuggingFace🤗 Hub: [ATCO2](https://hf.co/jlvdoorn/datasets/atco2-asr/), [ATCOSIM](https://hf.co/jlvdoorn/datasets/atcosim), and [ATCO2-ATCOSIM](https://hf.co/jlvdoorn/datasets/atco2-asr-atcosim/).

## Audio Info

The ```Audio Info.ipynb``` has some functions to calculate statistics (such as the total duration, number of files and sampling frequency) about the datasets.
