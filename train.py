__author__='thiagocastroferreira'

import nltk
from nltk.translate.bleu_score import corpus_bleu
nltk.download('punkt')
import os
import torch
from torch import optim

class Trainer:
    def __init__(self, model, trainloader, devloader, optimizer, epochs, \
        batch_status, device, write_path, early_stop=5, verbose=True, language='portuguese'):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_status = batch_status
        self.device = device
        self.early_stop = early_stop
        self.verbose = verbose
        self.trainloader = trainloader
        self.devloader = devloader
        self.write_path = write_path
        self.language = language
        if not os.path.exists(write_path):
            os.mkdir(write_path)
    
    def train(self):
        max_bleu, repeat = 0, 0
        for epoch in range(self.epochs):
            self.model.model.train()
            losses = []
            for batch_idx, inp in enumerate(self.trainloader):
                intents, texts = inp['X'], inp['y']
                self.optimizer.zero_grad()

                # generating
                output = self.model(intents, texts)

                # Calculate loss
                loss = output.loss
                losses.append(float(loss))

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                # Display
                if (batch_idx+1) % self.batch_status == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTotal Loss: {:.6f}'.format(epoch, \
                        batch_idx+1, len(self.trainloader), 100. * batch_idx / len(self.trainloader), 
                        float(loss), round(sum(losses) / len(losses), 5)))
            
            bleu = self.evaluate()
            print('BLEU: ', bleu)
            if bleu > max_bleu:
                self.model.model.save_pretrained(os.path.join(self.write_path, 'model'))
                max_bleu = bleu
                repeat = 0
                print('Saving best model...')
            else:
                repeat += 1
            
            if repeat == self.early_stop:
                break
    
    def evaluate(self):
        self.model.model.eval()
        y_pred, y_real = [], []
        for batch_idx, inp in enumerate(self.devloader):
            intents, texts = inp['X'], inp['y']

            # predict
            output = self.model(intents)
            
            y_pred.extend(output)
            y_real.extend(texts)

            # Display
            if (batch_idx+1) % self.batch_status == 0:
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
            
            if self.language != 'english':
                hyps.append(nltk.word_tokenize(snt_pred, language=self.language))
                refs.append([nltk.word_tokenize(y_real[i], language=self.language)])
            else:
                hyps.append(nltk.word_tokenize(snt_pred))
                refs.append([nltk.word_tokenize(y_real[i])])
        
        bleu = corpus_bleu(refs, hyps)
        return bleu