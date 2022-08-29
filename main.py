__author__='thiagocastroferreira'

import argparse
import botsdobem.load_data as botsdobem
import botsdobem.load_data_en as botsdobem_en
import webnlg.load_data as webnlg
import e2e.load_data as e2e
from models.bartgen import BARTGen
from models.bert import BERTGen
from models.gportuguesegen import GPorTugueseGen
from models.t5gen import T5Gen
from models.gpt2 import GPT2
from train import Trainer
from torch import optim
from torch.utils.data import DataLoader, Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("tokenizer", help="path to the tokenizer")
    parser.add_argument("model", help="path to the model")
    parser.add_argument("data", help="path to the data")
    parser.add_argument("epochs", help="number of epochs", type=int)
    parser.add_argument("learning_rate", help="learning rate", type=float)
    parser.add_argument("train_batch_size", help="batch size of training", type=int)
    parser.add_argument("early_stop", help="earling stop", type=int)
    parser.add_argument("max_length", help="maximum length to be processed by the network", type=int)
    parser.add_argument("write_path", help="path to write best model")
    parser.add_argument("language", help="language")
    parser.add_argument("--verbose", help="should display the loss?", action="store_true")
    parser.add_argument("--batch_status", help="display of loss", type=int)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    parser.add_argument("--describe_number_src", help="Describe numbers in the source?", action="store_true")
    parser.add_argument("--describe_number_trg", help="Describe numbers in the target?", action="store_true")
    parser.add_argument("--src_lang", help="source language of mBART tokenizer", default='pt_XX')
    parser.add_argument("--trg_lang", help="target language of mBART tokenizer", default='pt_XX')
    args = parser.parse_args()
    # settings
    learning_rate = args.learning_rate #1e-5
    epochs = args.epochs # 25
    train_batch_size = args.train_batch_size # 2
    batch_status = args.batch_status # 5
    early_stop =args.early_stop # 5
    language = args.language
    try:
        verbose = args.verbose # True
    except:
        verbose = False
    try:
        device = 'cuda:1' if args.cuda else 'cpu' # 'cuda'
    except:
        device = 'cpu'
    
    try:
        describe_number_src = True if args.describe_number_src else False
    except:
        describe_number_src = False
    try:
        describe_number_trg = True if args.describe_number_trg else False
    except:
        describe_number_trg = False
    write_path = args.write_path

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
        if language == 'english':
            traindata, devdata, testdata = botsdobem_en.load('synthetic', describe_number_src, describe_number_trg)
            dataset = botsdobem_en.NewsDataset(traindata)
            trainloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
        else:
            if 'original' in data:
                traindata, devdata = botsdobem.load('original', describe_number_src, describe_number_trg)
            else:
                traindata, devdata, testdata = botsdobem.load('synthetic', describe_number_src, describe_number_trg)

            dataset = botsdobem.NewsDataset(traindata)
            trainloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    elif 'webnlg' in data:
        traindata, devdata, testdata = webnlg.load()

        dataset = webnlg.NewsDataset(traindata)
        trainloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    elif 'e2e' in data:
        traindata, devdata, testdata = e2e.load()

        dataset = webnlg.NewsDataset(traindata)
        trainloader = DataLoader(dataset, batch_size=train_batch_size)
    else:
        raise Exception("Invalid dataset")

    # optimizer
    optimizer = optim.AdamW(generator.model.parameters(), lr=learning_rate)
    
    # trainer
    trainer = Trainer(generator, trainloader, devdata, optimizer, epochs, batch_status, device, write_path, early_stop, verbose, language)
    trainer.train()