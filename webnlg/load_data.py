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
            for triple in sorted(entry[key]['modifiedtripleset'], key=lambda x: (x['predicate'], x['subject'], x['object'])):
                subj, pred, obj = triple['subject'], triple['predicate'], triple['object']
                str_triple = ' '.join(['<subject>', subj, '<predicate>', pred, '<object>', obj])
                triples.append(str_triple)
            triples = '<triple>'.join(triple)
            inp = ' '.join([category, triples])

            for lexicalization in entry[key]['lexicalisations']:
                text = lexicalisation['lex']
                
                result.append({ 'X': inp, 'y': text })
    return result


def load():
    traindata = preprocess(json.load(open('webnlg/data_v2.1/train.json')))
    devdata = preprocess(json.load(open('webnlg/data_v2.1/dev.json')))
    testdata = preprocess(json.load(open('webnlg/data_v2.1/test.json')))

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