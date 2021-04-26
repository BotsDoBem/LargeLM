__author__='thiagocastroferreira'

import os
import argparse
import botsdobem.load_data as botsdobem
import webnlg.load_data as webnlg
import e2e.load_data as e2e
from models.bartgen import BARTGen
from models.bert import BERTGen
from models.gportuguesegen import GPorTugueseGen
from models.t5gen import T5Gen
from models.gpt2 import GPT2
from torch.utils.data import DataLoader, Dataset

class Inferencer:
    def __init__(self, model, dataloader, batch_status, device, write_dir, verbose=True, language='portuguese'):
        self.model = model
        self.batch_status = batch_status
        self.device = device
        self.verbose = verbose
        self.dataloader = dataloader
        self.write_dir = write_dir
        self.language = language
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
    
    
    def evaluate(self):
        self.model.model.eval()
        X, y_pred, y_real = [], [], []
        for batch_idx, inp in enumerate(self.dataloader):
            intents, texts = inp['X'], inp['y']

            # predict
            output = self.model(intents)
            
            X.extend(intents)
            y_pred.extend(output)
            y_real.extend(texts)

            # Display
            if (batch_idx+1) % self.batch_status == 0:
                print('Evaluation: [{}/{} ({:.0f}%)]'.format(batch_idx+1, \
                    len(self.dataloader), 100. * batch_idx / len(self.dataloader)))
        
        for i in range(len(intents)):
            path = os.path.join(self.write_dir, 'data.txt')
            with open(path, 'w') as f:
                f.write('\n'.join(X))
            
            path = os.path.join(self.write_dir, 'gold.txt')
            with open(path, 'w') as f:
                f.write('\n'.join(y_real))
            
            path = os.path.join(self.write_dir, 'output.txt')
            with open(path, 'w') as f:
                f.write('\n'.join(y_pred))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("tokenizer", help="path to the tokenizer")
    parser.add_argument("model", help="path to the model")
    parser.add_argument("data", help="path to the data")
    parser.add_argument("batch_size", help="batch size of test", type=int)
    parser.add_argument("max_length", help="maximum length to be processed by the network", type=int)
    parser.add_argument("write_dir", help="path to write results")
    parser.add_argument("language", help="language")
    parser.add_argument("--verbose", help="should display the loss?", action="store_true")
    parser.add_argument("--batch_status", help="display of loss", type=int)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    parser.add_argument("--src_lang", help="source language of mBART tokenizer", default='pt_XX')
    parser.add_argument("--trg_lang", help="target language of mBART tokenizer", default='pt_XX')
    args = parser.parse_args()
    # settings
    batch_size = args.batch_size # 2
    batch_status = args.batch_status # 5
    language = args.language
    try:
        verbose = args.verbose # True
    except:
        verbose = False
    try:
        device = 'cuda' if args.cuda else 'cpu' # 'cuda'
    except:
        device = 'cpu'
    write_dir = args.write_dir

    # model
    max_length = args.max_length # 128
    tokenizer_path = args.tokenizer # "facebook/mbart-large-50"
    model_path = args.model # "facebook/mbart-large-50"
    if 'mbart' in tokenizer_path:
        src_lang = args.src_lang
        trg_lang = args.trg_lang
        generator = BARTGen(tokenizer_path, model_path, max_length, device, True, src_lang, trg_lang)
    elif 'bart' in tokenizer_path:
        generator = BARTGen(tokenizer_path, model_path, max_length, device, False)
    elif 'bert' in tokenizer_path:
        generator = BERTGen(tokenizer_path, model_path, max_length, device)
    elif 'mt5' in tokenizer_path:
        generator = T5Gen(tokenizer_path, model_path, max_length, device, True)
    elif 't5' in tokenizer_path:
        generator = T5Gen(tokenizer_path, model_path, max_length, device, False)
    elif 'gpt2-small-portuguese' in tokenizer_path:
        generator = GPorTugueseGen(tokenizer_path, model_path, max_length, device)
    elif tokenizer_path == 'gpt2':
        generator = GPT2(tokenizer_path, model_path, max_length, device)
    else:
        raise Exception("Invalid model") 

    # data
    data = args.data
    if 'botsdobem' in data:
        traindata, devdata, testdata = botsdobem.load('synthetic')
            
        dataset = botsdobem.NewsDataset(testdata)
        dataloader = DataLoader(dataset, batch_size=batch_size)

    elif 'webnlg' in data:
        traindata, devdata, testdata = webnlg.load()

        dataset = botsdobem.NewsDataset(testdata)
        dataloader = DataLoader(dataset, batch_size=batch_size)
    elif 'e2e' in data:
        traindata, devdata, testdata = e2e.load()

        dataset = botsdobem.NewsDataset(testdata)
        dataloader = DataLoader(dataset, batch_size=batch_size)
    else:
        raise Exception("Invalid dataset")

    inf = Inferencer(generator, dataloader, batch_status, device, write_dir, verbose, language)
    inf.evaluate()