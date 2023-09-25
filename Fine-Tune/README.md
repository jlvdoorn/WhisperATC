# Fine-Tune

This folder is used for the fine-tuning of the models. It includes the following files:

- ```FineTuning-*.py``` is used to fine-tune blank Whisper on the * dataset.
- ```ft-*.sh``` is used to submit the fine-tuning of blank Whisper on the * dataset.
- ```ds_config.json``` is the [DeepSpeed](https://www.github.com/microsoft/deepspeed) configuration file that is used in order to fit the model on a single NVIDIA V100S 32GB GPU.
