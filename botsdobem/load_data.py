__author__='thiagocastroferreira'

import json 
import os
from random import shuffle
from sklearn.model_selection import train_test_split

DOMAINS = ['covid19', 'deforestation_daily', 'deforestation_month', 'amazon_fire', 'ibge_ipca']

DOMAIN2TOKEN = {
    'covid19': '[COVID19]',
    'deforestation_daily': '[DEFORESTATION_DAILY]',
    'deforestation_month': '[DEFORESTATION_MONTH]',
    'amazon_fire': '[AMAZON_FIRE]',
    'ibge_ipca': '[IBGE_IPCA]',
}

DOMAIN2PATH = {
    'covid19': 'covid19.json',
    'deforestation_daily': 'deter_daily.json',
    'deforestation_month': 'deter_month.json',
    'amazon_fire': 'amazon_fire.json',
    'ibge_ipca': 'ibge_ipca.json',
}

TEST_SPLIT=0.2

def format_intent(message, domain):
    attributes = sorted(message['attributes'].keys())
    attr_values = []
    for attr in attributes:
        value = str(message['attributes'][attr])
        # in case of numbers, normalize them
        if str(value.replace('.', '')).isnumeric() :
            value = float(value)
            # fix inconsistencies with percentages in covid and deforestation
            if attr == 'variation' and domain in ['covid19', 'deforestation_daily', 'deforestation_month']:
                value = float(value) * 100
            value = int(value) if value.is_integer() else round(value, 2)
            value = str(value)
      
        attr_values.append(attr + "=\"" +  value + "\"")
        
    intent = message['intent']
    str_msg = intent + '(' + ','.join(attr_values) + ')'
    return str_msg

def preprocess(data, domain):
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
                snt_intents = ' [SEP] '.join([format_intent(intent, domain) for intent in snt])
                prev = history[-1]
                inp = '[INTENTS] ' + snt_intents + ' [HISTORY] ' + prev
                snt_text = row['paragraphs'][pidx][sntidx]
                history.append(snt_text)

                result.append({ 'X': DOMAIN2TOKEN[domain] + ' ' + inp, 'y': snt_text })
    return result

def load(setting='original'):
    """
    Args:
        setting: 
            original - only human-annotated data
            synthetic - original + synthetic data
    """
    assert setting in ['original', 'synthetic']
    traindata, testdata = [], []
    for domain in DOMAINS:
        data = json.load(open(os.path.join('data', setting, DOMAIN2PATH[domain])))
        shuffle(data)
        size = int(len(data) * TEST_SPLIT)
        domain_traindata, domain_testdata = data[size:], data[:size]
        traindata.extend(preprocess(domain_traindata, domain))
        testdata.extend(preprocess(domain_testdata, domain))

    return traindata, testdata