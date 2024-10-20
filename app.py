from flask import Flask, request
from bot import Bot

app = Flask(__name__)
BOT = Bot()

@app.route('/messages', methods=['POST'])
def messages():
    user_question = request.json.get('message')
    if user_question:
        answer = BOT.handle_message(user_question)
        return {"answer": answer}
    return {"error": "Nenhuma pergunta recebida"}

if __name__ == '__main__':
    app.run(debug=True)
