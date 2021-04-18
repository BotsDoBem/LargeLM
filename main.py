__author__='thiagocastroferreira'

import argparse
import botsdobem.load_data as botsdobem
import webnlg.load_data as webnlg
import e2e.load_data as e2e
from models.bartgen import BARTGen
from models.bert import BERTGen
from models.gportuguesegen import GPorTugueseGen
from models.t5gen import T5Gen
from train import Trainer
from torch import optim
from torch.utils.data import DataLoader, Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("model", help="path to the model")
    parser.add_argument("data", help="path to the data")
    parser.add_argument("epochs", help="number of epochs", type=int)
    parser.add_argument("learning_rate", help="learning rate", type=float)
    parser.add_argument("train_batch_size", help="batch size of training", type=int)
    parser.add_argument("dev_batch_size", help="batch size of test", type=int)
    parser.add_argument("early_stop", help="earling stop", type=int)
    parser.add_argument("max_length", help="maximum length to be processed by the network", type=int)
    parser.add_argument("write_path", help="path to write best model")
    parser.add_argument("language", help="language")
    parser.add_argument("--verbose", help="should display the loss?", action="store_true")
    parser.add_argument("--batch_status", help="display of loss", type=int)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    parser.add_argument("--src_lang", help="source language of mBART tokenizer", default='pt_XX')
    parser.add_argument("--trg_lang", help="target language of mBART tokenizer", default='pt_XX')
    args = parser.parse_args()
    # settings
    learning_rate = args.learning_rate #1e-5
    epochs = args.epochs # 25
    train_batch_size = args.train_batch_size # 2
    dev_batch_size = args.dev_batch_size # 2
    batch_status = args.batch_status # 5
    early_stop =args.early_stop # 5
    language = args.language
    try:
        verbose = args.verbose # True
    except:
        verbose = False
    try:
        device = 'cuda' if args.cuda else 'cpu' # 'cuda'
    except:
        device = 'cpu'
    write_path = args.write_path

    # model
    max_length = args.max_length # 128
    path = args.model # "facebook/mbart-large-50"
    if 'mbart' in path:
        src_lang = args.src_lang
        trg_lang = args.trg_lang
        generator = BARTGen(path, max_length, device, True, src_lang, trg_lang)
    elif 'bart' in path:
        generator = BARTGen(path, max_length, device, False)
    elif 'bert' in path:
        generator = BERTGen(path, max_length, device)
    elif 'mt5' in path:
        generator = T5Gen(path, max_length, device, True)
    elif 't5' in path:
        generator = T5Gen(path, max_length, device, False)
    elif 'gpt2-small-portuguese' in path:
        generator = GPorTugueseGen(path, max_length, device)
    else:
        raise Exception("Invalid model") 

    # data
    data = args.data
    if 'botsdobem' in data:
        if 'original' in data:
            traindata, testdata = botsdobem.load('original')
        else:
            traindata, testdata = botsdobem.load('synthetic')
            
        dataset = botsdobem.NewsDataset(traindata)
        trainloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

        dataset = botsdobem.NewsDataset(testdata)
        testloader = DataLoader(dataset, batch_size=dev_batch_size, shuffle=True)
    elif 'webnlg' in data:
        traindata, devdata, testdata = webnlg.load()

        dataset = webnlg.NewsDataset(traindata)
        trainloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

        dataset = webnlg.NewsDataset(devdata)
        testloader = DataLoader(dataset, batch_size=dev_batch_size, shuffle=True)
    elif 'e2e' in data:
        traindata, devdata, testdata = e2e.load()

        dataset = webnlg.NewsDataset(traindata)
        trainloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

        dataset = webnlg.NewsDataset(devdata)
        testloader = DataLoader(dataset, batch_size=dev_batch_size, shuffle=True)
    else:
        raise Exception("Invalid dataset")

    # optimizer
    optimizer = optim.AdamW(generator.model.parameters(), lr=learning_rate)
    
    # trainer
    trainer = Trainer(generator, trainloader, testloader, optimizer, epochs, batch_status, device, write_path, early_stop, verbose, language)
    trainer.train()