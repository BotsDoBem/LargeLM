__author__='thiagocastroferreira'

import json 
import os
from random import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from num2words import num2words
import stanza
stanza.download('en') 

DOMAINS = ['deforestation_daily', 'deforestation_month']

DOMAIN2TOKEN = {
    'deforestation_daily': '[DEFORESTATION_DAILY]',
    'deforestation_month': '[DEFORESTATION_MONTH]'
}

DOMAIN2PATH = {
    'deforestation_daily': 'deter_daily.json',
    'deforestation_month': 'deter_month.json'
}

DOMAIN2PATH_SYNTHETIC = {
    'deforestation_daily': 'amazondaily',
    'deforestation_month': 'amazonmonth',
}

TEST_SPLIT=0.2

def format_intent(message, domain, describe_numbers=False):
    attributes = sorted(message['attributes'].keys())
    attr_values = []
    for attr in attributes:
        value = str(message['attributes'][attr])
        # in case of numbers, normalize them
        if str(value.replace('.', '')).isnumeric():
            value = float(value)
            # fix inconsistencies with percentages in covid and deforestation
            if attr == 'variation' and domain in ['deforestation_daily', 'deforestation_month']:
                value = float(value) * 100
            value = int(value) if value.is_integer() else round(value, 2)
            if describe_numbers:
                value = num2words(value, lang='en')
            else:
                value = str(value)
      
        attr_values.append(attr + "=\"" +  value + "\"")
        
    intent = message['intent']
    str_msg = intent + '(' + ','.join(attr_values) + ')'
    return str_msg

def format_value(value, language='pt-br'):
    value = '{:,.2f}'.format(float(value))
    value = value.split('.')
    if language == 'pt-br':
        value[0] = value[0].replace(',', '.')
        if value[1] == '00':
            return value[0]
        else:
            return ','.join(value)
    else:
        if value[1] == '00':
            return value[0]
        else:
            return '.'.join(value)

def format_text(sentence, describe_numbers=False, nlp=None):
    """
    rounding numbers in the sentence
    """
    doc = nlp(sentence)
    numbers = []
    for snt in doc.sentences:
        for token in snt.words:
            if token.upos == 'NUM':
                numbers.append(token.text)
    
    for number in numbers:
        if describe_numbers:
            try:
                desc = num2words(number, lang='en')
                sentence = sentence.replace(number, desc, 1)
            except:
                pass
    if describe_numbers:
        sentence = sentence.replace('%', ' percent')
    return sentence

def preprocess(data, domain, setting, describe_number_src=False, describe_number_trg=False, nlp=None):
    """
    Args:
        data: grammar
        domain: domain of the grammar
    """
    result = []
    for row in data:
        history = []
        for pidx, p in enumerate(row['structure']):
            history.append('[PARAGRAPH]')
            for sntidx, snt in enumerate(p):
                snt = sorted(snt, key=lambda x: x['intent'])
                snt_intents = ' [SEP] '.join([format_intent(intent, domain, describe_number_src) for intent in snt])
                prev = history[-1]
                inp = '[INTENTS] ' + snt_intents + ' [HISTORY] ' + prev
                if setting == 'original':
                    snt_text = format_text(row['paragraphs'][pidx][sntidx], describe_number_trg, nlp)
                else:
                    if describe_number_trg:
                        snt_text = format_text(row['paragraphs'][pidx][sntidx], describe_number_trg, nlp)
                    else:
                        snt_text = row['paragraphs'][pidx][sntidx]
                history.append(snt_text)

                result.append({ 'X': DOMAIN2TOKEN[domain] + ' ' + inp, 'y': snt_text })
    return result

def load(setting='synthetic', describe_number_src=False, describe_number_trg=False):
    """
    Args:
        setting: 
            original - only human-annotated data
            synthetic - original + synthetic data
    """
    assert setting in ['synthetic']
    nlp = None
    if describe_number_trg:
        nlp = stanza.Pipeline('en')
    
    traindata, devdata, testdata = [], [], []
    for domain in DOMAINS:
        if domain != 'ibge_ipca':
            print(domain)
            path = os.path.join('botsdobem/data/en', setting, DOMAIN2PATH_SYNTHETIC[domain])
            domain_traindata = json.load(open(os.path.join(path, 'traindata.json')))
            domain_devdata = json.load(open(os.path.join(path, 'devdata.json')))
            domain_testdata = json.load(open(os.path.join(path, 'testdata.json')))

            traindata.extend(preprocess(domain_traindata, domain, setting, describe_number_src, describe_number_trg, nlp))
            devdata.extend(preprocess(domain_devdata, domain, setting, describe_number_src, describe_number_trg, nlp))
            testdata.extend(preprocess(domain_testdata, domain, setting, describe_number_src, describe_number_trg, nlp))
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