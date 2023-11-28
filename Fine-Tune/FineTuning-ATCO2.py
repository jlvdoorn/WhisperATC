# %% [markdown]
# # Fine Tuning Testing

# %%
mdl = 'openai/whisper-large-v3'
dts = 'jlvdoorn/atco2-asr'

opd = './' + mdl.split('/')[-1] + '-' + dts.split('/')[-1]
print('Training Model : {}'.format(mdl))
print('On Dataset     : {}'.format(dts))
print('Output Dir.    : {}'.format(opd))

# %% [markdown]
# ### Initializing HuggingFace

# %%
# import os

# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"

# %%
# from huggingface_hub import notebook_login
# notebook_login()

# %% [markdown]
# ### Loading ATCO2-ASR Dataset

# %%
from datasets import load_dataset, DatasetDict

dataset = DatasetDict()

dataset['train'] = load_dataset(dts, split="train")
dataset['validation']  = load_dataset(dts, split="validation")
print(dataset)

# %%
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(mdl)

# %%
from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained(mdl, language="English", task="transcribe")

# %%
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(mdl, language="English", task="transcribe")

# %%
from datasets import Audio

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# %%
# Dataset should now contain at least 'audio' and 'text' columns
dataset

# %%
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)

# %%
# Dataset should now contain 'input_features' and 'labels'
dataset

# %%
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# %%
import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# %%
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(mdl)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# %%
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=opd,  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # Make sure that [gradient_accumulation_steps] * [Num of GPUs] = 64 (16 only for ATCO2)
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=2800,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    deepspeed="ds_config.json",
)

# %%
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
torch.cuda.empty_cache()

# %%
import transformers
transformers.logging.set_verbosity_info()
trainer.train()#resume_from_checkpoint=True)
trainer.save_model(opd)

# %%
kwargs = {
    "dataset_tags": dts.split('/')[-1],
    "dataset": "ATCO2-ASR",  # a 'pretty' name for the training dataset
    "dataset_args": "config: en, split: train",
    "language": "en",
    "model_name": "Whisper Large v2 - ATCO2-ASR",  # a 'pretty' name for your model
    "finetuned_from": mdl,
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}
trainer.push_to_hub()#**kwargs)


