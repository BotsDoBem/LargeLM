__author__='thiagocastroferreira'

import nltk
nltk.download('punkt')
import torch
from torch import optim
from load_data import load, NewsDataset
from generator import Generator

class Trainer:
    def __init__(self, model, trainloader, devloader, optimizer, epochs, \
        train_batch_size, dev_batch_size, batch_status, device, early_stop=5, verbose=True):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch_status = batch_status
        self.device = device
        self.early_stop = early_stop
        self.verbose = verbose
        self.trainloader = trainloader
        self.devloader = devloader
        self.model.to(device)
    
    def train(self):
        max_bleu, repeat = 0, 0
        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            for batch_idx, inp in enumerate(self.trainloader):
                intents, texts = inp['X'], inp['y']
                self.optimizer.zero_grad()

                # generating
                output = model(intents, texts)

                # Calculate loss
                loss = output.loss
                losses.append(float(loss))

                # Backpropagation
                loss.backward()
                optimizer.step()

                # Display
                if (batch_idx+1) % self.batch_status == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTotal Loss: {:.6f}'.format(epoch, \
                        batch_idx+1, len(self.trainloader), 100. * batch_idx / len(self.trainloader), 
                        float(loss), round(sum(losses) / len(losses), 5)))
            
            bleu = self.evaluate()
            if bleu > max_bleu:
                max_bleu = bleu
                repeat = 0
            else:
                repeat += 1
            
            if repeat == early_stop:
                break
    
    def evaluate(self):
        self.model.eval()
        y_pred, y_real = [], []
        for batch_idx, inp in enumerate(self.devloader):
            intents, texts = inp['X'], inp['y']

            # predict
            output = model(intents)
            
            y_pred.extend(output)
            y_real.append(texts)

            # Display
            if (batch_idx+1) % batch_status == 0:
                print('Evaluation: [{}/{} ({:.0f}%)]'.format(batch_idx+1, \
                    len(self.devloader), 100. * batch_idx / len(self.devloader)))
        
        results = []
        hyps, refs = [], []
        for i, snt_pred in enumerate(y_pred):
            results.append(snt_pred)
            if i < 20 and self.verbose:
                print('Real: ', y_real[i])
                print('Pred: ', snt_pred)
                print()
            
            hyps.append(nltk.word_tokenize(snt_pred, language='portuguese'))
            refs.append([nltk.word_tokenize(y_real[i], language='portuguese')])
        
        bleu = corpus_bleu(refs, hyps)
        return bleu

if __name__ == '__main__':
    # settings
    learning_rate = 1e-5
    epochs = 5
    train_batch_size = 2
    dev_batch_size = 2
    batch_status=5
    early_stop=5
    verbose=True
    device = 'cuda'

    # model
    path = "facebook/mbart-large-50"
    src_lang, trg_lang = "pt_XX", "pt_XX"
    max_length = 128
    generator = Generator(path, src_lang, trg_lang, max_length, device)

    # data
    traindata, testdata = load_data.load()
    dataset = NewsDataset(traindata)
    trainloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

    dataset = NewsDataset(testdata)
    testloader = DataLoader(dataset, batch_size=dev_batch_size, shuffle=True)

    # optimizer
    optimizer = optim.Adam(generator.model.parameters(), lr=learning_rate)
    
    # trainer
    trainer = Trainer(model, trainloader, testloader, optimizer, epochs, train_batch_size, dev_batch_size, batch_status, device, early_stop=5, verbose=True)
    trainer.train()