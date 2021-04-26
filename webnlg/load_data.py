__author__='thiagocastroferreira'

import json
import os
from torch.utils.data import DataLoader, Dataset

def preprocess(entryset):
    result = []
    for entry in entryset['entries']:
        for key in entry:
            category = entry[key]['category']

            triples = []
            for triple in sorted(entry[key]['modifiedtripleset'], key=lambda x: (x['property'], x['subject'], x['object'])):
                subj, pred, obj = triple['subject'], triple['property'], triple['object']
                str_triple = ' '.join(['<subject>', subj, '<predicate>', pred, '<object>', obj])
                triples.append(str_triple)
            triples = ' <triple> '.join(triples)
            inp = ' '.join([category, triples])

            for lexicalization in entry[key]['lexicalisations']:
                text = lexicalization['lex']
                
                result.append({ 'X': inp, 'y': text })
    return result

def preprocess_evaluation(entryset):
    result = {}
    for entry in entryset['entries']:
        for key in entry:
            category = entry[key]['category']

            triples = []
            for triple in sorted(entry[key]['modifiedtripleset'], key=lambda x: (x['property'], x['subject'], x['object'])):
                subj, pred, obj = triple['subject'], triple['property'], triple['object']
                str_triple = ' '.join(['<subject>', subj, '<predicate>', pred, '<object>', obj])
                triples.append(str_triple)
            triples = ' <triple> '.join(triples)
            inp = ' '.join([category, triples])
            if inp not in result:
                result[inp] = { 'X': inp, 'y': [] }

            for lexicalization in entry[key]['lexicalisations']:
                text = lexicalization['lex']
                
                result[inp]['y'].append(text)
    return result.values()


def load():
    traindata = preprocess(json.load(open('webnlg/data_v2.1/train.json')))
    devdata = preprocess_evaluation(json.load(open('webnlg/data_v2.1/dev.json')))
    testdata = preprocess_evaluation(json.load(open('webnlg/data_v2.1/test.json')))

    return traindata, devdata, testdata

class NewsDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (string): data
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]