# Evaluate

This folder is used for the evaluation of the models, both the baseline and the fine-tuned models. It includes the following files:

- ```Evaluate-Blank.py``` is used to evaluate the baseline performance of the models on the ATCO2 and ATCOSIM datasets.
- ```Evaluate-FineTuned.py``` is used to evaluate the fine-tuned performance of the models on the ATCO2 and ATCOSIM datasets.
- ```Normalizer.py``` is the constructed normalizer used for all the performance asessments.

## Evaluate-*.py

In the Evaluate-Blank.py file, the model is set to Whisper Large v2. The dataset and split can be set in the line 5 and 7 respectively. In the Evaluate-FineTuned.py file, the model, dataset and split can be set in lines 5-7.
