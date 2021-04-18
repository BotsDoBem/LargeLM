__author__='thiagocastroferreira'

import csv
import os
from torch.utils.data import DataLoader, Dataset

def read(path):
    data = []
    with open(path) as f:
        reader = csv.reader(f, delimiter=',', quotechar='\"')
        for row in reader:
            data.append({ 'X': row[0], 'y': row[1] })
    return data

def load():
    traindata = read('e2e/data/trainset.csv')
    devdata = read('e2e/data/devset.csv')
    testdata = read('e2e/data/testset_w_refs.csv')

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