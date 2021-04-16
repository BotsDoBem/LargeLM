__author__='thiagocastroferreira'

import torch
import torch.nn as nn

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class Generator:
    def __init__(self, path, src_lang, trg_lang, max_length, device):
        self.tokenizer = MBart50TokenizerFast.from_pretrained(path, src_lang=src_lang, tgt_lang=trg_lang)
        self.model = MBartForConditionalGeneration.from_pretrained(path).to(device)
        self.device = device
        self.max_length = max_length

    def forward(intents, texts=None):
        # tokenize
        model_inputs = self.tokenizer(intents, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        # Predict
        if text:
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").input_ids.to(self.device)
            # Predict
            output = self.model(**model_inputs, labels=labels) # forward pass
        else:
            generated_ids = self.model.generate(**model_inputs)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return output