import random, json, math, string, re
from NB import NaiveBayes
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = NaiveBayes()
model.load_model('model')
file = open('messages.json', 'r')
answer_cs = json.loads(file.read())
file.close()

def getAnswer(question):
    id_answer = model.predict([question])[0]
    if len(id_answer) > 1:
        answer = 'Maaf saya kurang mengerti\nMungkin ini jawaban yang anda cari:'
        for a in range(len(id_answer)):
            answer += '\n{}. {}'.format(a+1, answer_cs[str(id_answer[a])][0])
    else:
        answer = answer_cs[str(id_answer[0])][0]
    return answer

@app.route('/', methods=['GET', 'POST'])
def API():
    if request.method == "GET":
        return render_template('home.html')
    elif request.method == "POST":
        question = request.form['question']
        answer  = getAnswer(question)
        return answer

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6598', threaded=True)