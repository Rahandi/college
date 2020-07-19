import time, json
import gcloud
from NB import NaiveBayes

model = NaiveBayes()
model.load_model('model')
file = open('messages.json', 'r')
answer_cs = json.loads(file.read())
file.close()

while True:
	question = input('inputan: ')
	if question == 'quit':
		exit()
	now = time.time()
	print(gcloud.getAnswer(gcloud.getPrediction(question)[0]))
	print('cnn: ' + str(time.time()-now))
	now = time.time()
	print(answer_cs[str(model.predict([question])[0])][0])
	print('naive bayes: ' + str(time.time()-now))
