import json
import math
import matplotlib.pyplot as plt
plt.style.use('ggplot')

file = open('master_hasil_skenario.txt', 'r')
master = json.loads(file.read())
file.close()

master_video = master['video']
video_acc = []
for key, value in master_video.items():
    video_acc.append(value['accuracy'] * 100)

plt.bar(range(len(video_acc)), video_acc)
plt.xlabel('Kondisi')
plt.ylabel('Akurasi')
plt.xticks(range(len(video_acc)), ['siang', 'malam'])

for i, v in enumerate(video_acc):
    plt.text(i-.125, 1, '{:.2f}'.format(v))

plt.show()