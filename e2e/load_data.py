__author__='thiagocastroferreira'

import csv

def read(path):
    data = []
    with open(path) as f:
        reader = csv.reader(f, delimiter=',', quotechar='\"')
        for row in reader:
            data.append({ 'X': row[0], 'y': row[1] })
    return data

def load():
    traindata = read('trainset.csv')
    devdata = read('devset.csv')
    testdata = read('testset_w_refs.csv')

    return traindata, devdata, testdata