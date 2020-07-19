import pandas as pd
import json
import math
import time
from NB import NaiveBayes
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

def test_train_sama(save_to=False):
    data = pd.read_csv('updated.csv')
    X = data['kalimat'].values.tolist()
    Y = data['intent_num'].values.tolist()
    model = NaiveBayes()
    model.fit(X, Y)
    if save_to != False:
        model.save_model(save_to)
    predict = model.predict(X)
    print('Accuracy: {0:.2f}%'.format(100*accuracy_score(Y, predict)))
    print('Precision: {0:.2f}%'.format(100*precision_score(Y, predict, average='macro')))
    print('Recall: {0:.2f}%'.format(100*recall_score(Y, predict, average='macro')))

def test_train_dibagi(iterasi, best_metric='accuracy', save_to=False):
    data = pd.read_csv('updated.csv')
    X = data['kalimat'].values.tolist()
    Y = data['intent_num'].values.tolist()
    best = {
        'accuracy':0,
        'precision':0,
        'recall':0
    }
    for a in range(iterasi):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        model=NaiveBayes()
        model.fit(x_train, y_train)
        predict = model.predict(x_test)
        current = {
            'accuracy':100*accuracy_score(y_test, predict),
            'precision':100*precision_score(y_test, predict, average='macro'),
            'recall':100*recall_score(y_test, predict, average='macro')
        }
        print('Iterasi: {}'.format(a+1))
        print('Accuracy: {0:.2f}%'.format(current['accuracy']))
        print('Precision: {0:.2f}%'.format(current['precision']))
        print('Recall: {0:.2f}%'.format(current['recall']))
        if current[best_metric] > best[best_metric]:
            best['accuracy'] = current['accuracy']
            best['precision'] = current['precision']
            best['recall'] = current['recall']
            if save_to != False:
                model.save_model(save_to)
    print('best\n' + str(best))

def text_input():
    file = open('messages.json', 'r')
    answer = json.loads(file.read())
    data = pd.read_csv('updated.csv')
    X = data['kalimat'].values.tolist()
    Y = data['intent_num'].values.tolist()
    model = NaiveBayes()
    model.fit(X, Y)
    while True:
        inputan = input('masukan input: ')
        if inputan == 'exit':
            exit()
        predict = model.single_predict([inputan])
        print(answer[str(predict[0])][0])

def train_then_test(test_file):
    file = open('messages.json', 'r')
    answer = json.loads(file.read())
    train = pd.read_csv('updated.csv')
    test = file.read('test_file', 'r')
    test = test.read()
    x_train = train['kalimat'].values.tolist()
    y_train = train['intent_num'].values.tolist()
    model = NaiveBayes()
    model.fit(x_train, y_train)
    jawaban = model.predict(test)
    for a in jawaban:
        print(answer[str(a)][0])

def coba_print():
    data = pd.read_csv('updated.csv')
    X = data['kalimat'].values.tolist()
    Y = data['intent_num'].values.tolist()
    model = NaiveBayes()
    model.fit(X, Y)
    print(model)

def cross_val():
    data = pd.read_csv('updated.csv')
    X = data['kalimat'].values.tolist()
    Y = data['intent_num'].values.tolist()
    kfold = 10
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    skf = StratifiedKFold(n_splits = kfold)
    model = NaiveBayes()
    for train_index, test_index in skf.split(X, Y):
        for a in range(len(train_index)):
            X_train.append(X[train_index[a]])
            y_train.append(Y[train_index[a]])
        for a in range(len(test_index)):
            X_test.append(X[test_index[a]])
            y_test.append(Y[test_index[a]])
        model.fit(X_train, y_train)
        model_result = model.single_predict(X_test)
        cr = classification_report(model_result, y_test)
        print(cr)

def train_test():
    train_data = pd.read_csv('updated.csv')
    test_data = pd.read_csv('testing.csv', delimiter=',', header=None)
    dataTrain = train_data['kalimat'].values.tolist()
    classTrain = train_data['intent_num'].values.tolist()
    dataTest = test_data[0]
    model = NaiveBayes()
    model.fit(dataTrain, classTrain)
    now = time.time()
    model_result = model.predict([dataTest[0]])
    print('1 pertanyaan: ' + str(time.time()-now))
    now = time.time()
    model_result = model.predict(dataTest)
    print('100 pertanyaan: ' + str(time.time()-now))
    for a in range(len(model_result)):
        print(a, model_result[a])


if __name__ == '__main__':
    test_train_sama(save_to='server/model')
    # test_train_dibagi(100, 'precision')
    # coba_print()
    # text_input()
    # cross_val()
    # train_test()