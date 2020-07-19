import pandas as pd
import json
from NB import NaiveBayes
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold

def test_train_sama(save_to=False):
    data = pd.read_csv('updated.csv')
    X = data['kalimat'].values.tolist()
    Y = data['intent_num'].values.tolist()
    model = NaiveBayes()
    model.fit(X, Y)
    if save_to != False:
        model.save_model(save_to)
    # predict = model.predict(X)
    print(model.num_messages)
    print('\n')
    print(model.log_class_priors)
    print('\n')
    print(model.word_counts)
    # print('Accuracy: {0:.2f}%'.format(100*accuracy_score(Y, predict)))
    # print('Precision: {0:.2f}%'.format(100*precision_score(Y, predict, average='macro')))
    # print('Recall: {0:.2f}%'.format(100*recall_score(Y, predict, average='macro')))

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
        predict = model.predict([inputan])
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
    model = NaiveBayes()
    kfold = 10
    

if __name__ == '__main__':
    # test_train_sama(save_to='server/model')
    # test_train_dibagi(100, 'precision')
    # coba_print()
    # text_input()
    cross_val()