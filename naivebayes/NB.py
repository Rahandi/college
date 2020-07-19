import pandas as pd
import os, re, string, math, json, pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import copy

class NaiveBayes:
    def __init__(self):
        self.num_messages = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()
        factory = StemmerFactory()
        factory_stopword = StopWordRemoverFactory()
        self.stopword = factory_stopword.create_stop_word_remover()
        self.stemmer = factory.create_stemmer()

    def __str__(self):
        return str('num_messages: {}\n\nlog_class_priors: {}\n\nword_counts: {}\n\nvocab: {}'.format(self.num_messages, self.log_class_priors, self.word_counts, self.vocab))

    def load_model(self, model_file):
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        self.num_messages = data['num_messages']
        self.log_class_priors = data['log_class_priors']
        self.word_counts = data['word_counts']
        self.vocab = data['vocab']

    def clean(self, s):
        translator = str.maketrans("","", string.punctuation)
        return s.translate(translator)

    def stemming(self, word):
        return self.stemmer.stem(word)

    def tokenize(self, text):
        stemmed = []
        text = self.clean(text).lower()
        # text = self.stopword.remove(text)
        splitted = text.split()
        stemmed = [self.stemming(a) for a in splitted]
        return stemmed

    def get_word_counts(self, words):
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts

    def fit(self, X, Y):
        self.num_messages = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()

        n = len(X)
        for a in range(n):
            self.word_counts[Y[a]] = {}
            self.num_messages[Y[a]] = self.num_messages.get(Y[a], 0) + 1
        
        for b in self.num_messages.items():
            self.log_class_priors[b[0]] = math.log(self.num_messages[b[0]]/n)

        for x, y in zip(X, Y):
            counts = self.get_word_counts(self.tokenize(x))
            for word, count in counts.items():
                if word not in self.vocab:
                    self.vocab.add(word)
                if word not in self.word_counts[y]:
                    self.word_counts[y][word] = 0.0
                self.word_counts[y][word] += count

    def bubblesort(self, lists, id):
    # Swap the elements to arrange in order
        for iter_num in range(len(lists)-1,0,-1):
            for idx in lists.keys():
                next = int(idx)+1
                # next = str(next)
                if next not in lists:
                    # print(next)
                    continue
                if lists[idx]<lists[next]:
                    temp = lists[idx]
                    temp1 = id[idx]
                    lists[idx] = lists[next]
                    id[idx] = id[next]
                    lists[next] = temp
                    id[next] = temp1
        return lists, id

    def predict(self, X):
        result = []
        score = {}
        for x in X:
            for a in range(1, 27):
                score[a] = 0
            counts = self.get_word_counts(self.tokenize(x))
            for word, _ in counts.items():
                if word not in self.vocab: continue
                for b in range(1, len(score)+1):
                    log_w = math.log((self.word_counts[b].get(word, 0.0)+1)/(self.num_messages[b] + len(self.vocab)))
                    score[b] += log_w
            for b in range(1, len(score)+1):
                score[b] += self.log_class_priors[b]
            id = {}
            lists = {}
            lists = copy.deepcopy(score)
            for x in range(len(lists)):
                num = x+1
                id[num] = num
            value, key = self.bubblesort(lists, id)
            answer = []
            thresh = value[1] - 0.25
            for i in range(len(value)):
                num = i+1
                if value[num] > thresh:
                    answer.append(key[num])
            if value[1] > -3:
                answer = [27]
            result.append(answer)
        return result

    def single_predict(self, X):
        result = []
        score = {}
        for x in X:
            for a in range(1, 27):
                score[a] = 0
            counts = self.get_word_counts(self.tokenize(x))
            for word, _ in counts.items():
                if word not in self.vocab: 
                    continue
                for b in range(1, len(score)+1):
                    log_w = math.log((self.word_counts[b].get(word, 0.0)+1)/(self.num_messages[b] + len(self.vocab)))
                    score[b] += log_w
            for b in range(1, len(score)+1):
                score[b] += self.log_class_priors[b]
            id = {}
            lists = {}
            lists = copy.deepcopy(score)
            for x in range(len(lists)):
                num = x+1
                id[num] = num
            value, key = self.bubblesort(lists, id)
            if value[1] > -3:
                result.append(27)
            else:
                result.append(key[1])
        return result

    def save_model(self, save_to):
        tobe = {
            'num_messages':self.num_messages,
            'log_class_priors':self.log_class_priors,
            'word_counts':self.word_counts,
            'vocab':self.vocab
        }
        with open(save_to, 'wb') as f:
            pickle.dump(tobe, f, pickle.HIGHEST_PROTOCOL)