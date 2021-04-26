__author__='thiagocastroferreira'

import nltk
from nltk.translate.bleu_score import corpus_bleu
nltk.download('punkt')
import os
import torch
from torch import optim

class Trainer:
    def __init__(self, model, trainloader, devdata, optimizer, epochs, \
        batch_status, device, write_path, early_stop=5, verbose=True, language='portuguese'):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_status = batch_status
        self.device = device
        self.early_stop = early_stop
        self.verbose = verbose
        self.trainloader = trainloader
        self.devdata = devdata
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
        results = {}
        for batch_idx, inp in enumerate(self.devdata):
            intent, text = inp['X'], inp['y']
            if intent not in results:
                results[intent] = { 'hyp': '', 'refs': [] }
                # predict
                output = self.model([intent])
                results[intent]['hyp'] = output[0]

                # Display
                if (batch_idx+1) % self.batch_status == 0:
                    print('Evaluation: [{}/{} ({:.0f}%)]'.format(batch_idx+1, \
                        len(self.devdata), 100. * batch_idx / len(self.devdata)))
            
            results[intent]['refs'].append(text)
        
        hyps, refs = [], []
        for i, intent in enumerate(results.keys()):
            if i < 20 and self.verbose:
                print('Real: ', results[intent]['refs'][0])
                print('Pred: ', results[intent]['hyp'])
                print()
            
            if self.language != 'english':
                hyps.append(nltk.word_tokenize(results[intent]['hyp'], language=self.language))
                refs.append([nltk.word_tokenize(ref, language=self.language) for ref in results[intent]['refs']])
            else:
                hyps.append(nltk.word_tokenize(results[intent]['hyp']))
                refs.append([nltk.word_tokenize(ref) for ref in results[intent]['refs']])
        
        bleu = corpus_bleu(refs, hyps)
        return bleu