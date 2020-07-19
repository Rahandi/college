import json, requests, os, time, errno, sys, random
from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

app = Flask(__name__)

line_bot_api = LineBotApi('Gk5RoraVjZiqmyumktvYmbYXiAizwJ6bM8avSat0BWOlkiRr54nU9m+zpZEHKGrGV/ccjXkSU8ELOKOUzdSaVJzU2qhSf6HUZOqrgIEPQFYe8CtNzhO6KW7nQy/jTc6xfkvjjOFSUsydrjl+qFe9hAdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('7395664710bb5c57df29078dae9870e6')

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@app.route("/test", methods=['GET'])
def test():
    return "sip"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    if event.message.text == 'test':
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="mantab"))
    else:
        line_bot_api.reply_message(event.reply_token,TextSendMessage(text=ask_api(event.message.text)))

def ask_api(quest):
    try:
        url = 'http://13.76.0.42:6598/'
        payload = {'question':quest}
        resp = ''
        while resp == '':
            try:
                session = requests.Session()
                req = session.post(url, data=payload, timeout=15)
                resp = str(req.text)
            except Exception as e:
                time.sleep(1)
                continue
        return resp
    except Exception as e:
        return e

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)