# WhisperATC

Applying Large-Scale Weakly-Supervised Automatic Speech Recognition to Air Traffic Control

## Data Preparation

The ```CreateDataset``` module contains the necessary files to create and upload datasets to huggingface. A random split is made in the data where 80% will be used for training and 20% for validaiton. Currently the following datasets are available:

- [ATCO2-ASR](https://www.doi.org/10.57967/hf/1377)
- [ATCOSIM](https://www.doi.org/10.57967/hf/1378)
- [ATCO2-ASR-ATCOSIM](https://www.doi.org/10.57967/hf/1379)

However, the latter will only be used for training purposed and not for evaulation as it does not form a benchmark.

## Prompt Testing

The ```PromptTesting``` module contains the files used for the iterative experiments on prompting and normalization.

## Evaluation

Two scripts are available for evaluating the models. One is for evaluating the blank models and the second is for evaulating the fine-tuned models. The fine-tuned model weights will be converted into the whisper format in order to be used for inference. This folder also contains the normalization script.

## Fine-Tuning

The fine-tuning scripts are created to form a modular way of fine-tuning the blank Whisper models on the created datasets. The models will automatically be uploaded into the huggingface format. The fine-tuning relies on the ```deepspeed``` package. Currently the following fine-tuned models are available:

### Whisper Large V3

- [Whisper Large v3 - ATCO2](https://www.doi.org/10.57967/hf/1386)
- [Whisper Large v3 - ATCOSIM](https://www.doi.org/10.57967/hf/1387)
- [Whisper Large v3 - ATCO2 ATCOSIM](https://www.doi.org/10.57967/hf/1388)

### Whisper Large V2

- [Whisper Large v2 - ATCO2](https://www.doi.org/10.57967/hf/1376)
- [Whisper Large v2 - ATCOSIM](https://www.doi.org/10.57967/hf/1374)
- [Whisper Large v2 - ATCO2 ATCOSIM](https://www.doi.org/10.57967/hf/1375)

## HuggingFace🤗

All the datasets and models are available on the [HuggingFace🤗 Hub](https://huggingface.co/jlvdoorn).

## Demo

An interactive demo can be found on [HuggingFace🤗 Spaces](https://jlvdoorn-whisperatc.hf.space/).

## Paper

The paper can be found [here](http://resolver.tudelft.nl/uuid:8aa780bf-47b6-4f81-b112-29e23bc06a7d).

## License

All code is licensed under the LGPL-3.0 license. See the [LICENSE](LICENSE.txt) file for details.
In this repository we use [Whisper](https://www.github.com/openai/whisper) which is licensed under the [MIT License](https://github.com/openai/whisper/blob/main/LICENSE).

## Citation

If you use this code in your work, please cite the accompanying paper:

<!-- ```[bibtex]
@article{van2021applying,
  title={Applying Large-Scale Weakly-Supervised Automatic Speech Recognition to Air Traffic Control},
  author={van Doorn, Jeroen and van der Veen, Joris and van der Goot, Rob and van der Heijden, Ferdinand},
  journal={arXiv preprint arXiv:2109.14882},
  year={2021}
}
``` -->

The models can be cited as follows:

```[bibtex]
@misc {wlv3-atco2-asr,
  author       = { {J.L.P.M. van Doorn} },
  title        = { whisper-large-v3-atco2-asr },
  year         = 2023,
  url          = { https://huggingface.co/jlvdoorn/whisper-large-v3-atco2-asr },
  doi          = { 10.57967/hf/1386 },
  publisher    = { Hugging Face }
},

@misc {wlv3-atcosim,
  author       = { {J.L.P.M. van Doorn} },
  title        = { whisper-large-v3-atcosim },
  year         = 2023,
  url          = { https://huggingface.co/jlvdoorn/whisper-large-v3-atcosim },
  doi          = { 10.57967/hf/1387 },
  publisher    = { Hugging Face }
},

@misc {wlv3-atco2-asr-atcosim,
  author       = { {J.L.P.M. van Doorn} },
  title        = { whisper-large-v3-atco2-asr-atcosim },
  year         = 2023,
  url          = { https://huggingface.co/jlvdoorn/whisper-large-v3-atco2-asr-atcosim },
  doi          = { 10.57967/hf/1388 },
  publisher    = { Hugging Face }
},

@misc {wlv2-atco2-asr,
  author       = { {J.L.P.M. van Doorn} },
  title        = { whisper-large-v2-atco2-asr },
  year         = 2023,
  url          = { https://huggingface.co/jlvdoorn/whisper-large-v2-atco2-asr },
  doi          = { 10.57967/hf/1376 },
  publisher    = { Hugging Face }
},

@misc {wlv2-atcosim,
  author       = { {J.L.P.M. van Doorn} },
  title        = { whisper-large-v2-atcosim },
  year         = 2023,
  url          = { https://huggingface.co/jlvdoorn/whisper-large-v2-atcosim },
  doi          = { 10.57967/hf/1374 },
  publisher    = { Hugging Face }
},

@misc {wlv2-atco2-asr-atcosim,
  author       = { {J.L.P.M. van Doorn} },
  title        = { whisper-large-v2-atco2-asr-atcosim },
  year         = 2023,
  url          = { https://huggingface.co/jlvdoorn/whisper-large-v2-atco2-asr-atcosim },
  doi          = { 10.57967/hf/1375 },
  publisher    = { Hugging Face }
}
```

The datasets can be cited as follows:
  
```[bibtex]
@misc {atco2-asr,
  author       = { {J.L.P.M. van Doorn} },
  title        = { atco2-asr },
  year         = 2023,
  url          = { https://huggingface.co/datasets/jlvdoorn/atco2-asr },
  doi          = { 10.57967/hf/1377 },
  publisher    = { Hugging Face }
},

@misc {atcosim,
  author       = { {J.L.P.M. van Doorn} },
  title        = { atcosim },
  year         = 2023,
  url          = { https://huggingface.co/datasets/jlvdoorn/atcosim },
  doi          = { 10.57967/hf/1378 },
  publisher    = { Hugging Face }
},

@misc {atco2-asr-atcosim,
  author       = { {J.L.P.M. van Doorn} },
  title        = { atco2-asr-atcosim },
  year         = 2023,
  url          = { https://huggingface.co/datasets/jlvdoorn/atco2-asr-atcosim },
  doi          = { 10.57967/hf/1379 },
  publisher    = { Hugging Face }
}
```
