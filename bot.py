from nlp_service import NLPService
from dialogflow_service import DialogflowService

class Bot:
    def __init__(self):
        self.nlp_service = NLPService()
        self.dialogflow_service = DialogflowService()

    def handle_message(self, user_message):
        # Se a mensagem começa com 'dialogflow', a resposta virá do Dialogflow
        if user_message.startswith("dialogflow;"):
            # Remove o comando e usa o Dialogflow para gerar a resposta
            command, message = user_message.split(';', 1)
            return self.dialogflow_service.detect_intent_text(message)
        # Se for um comando de treinamento
        elif user_message.startswith("treinarmodelo;"):
            parts = user_message.split(';')
            if len(parts) == 3:
                question = parts[1].strip()
                answer = parts[2].strip()
                return self.nlp_service.train_with_input(question, answer)
            else:
                return "Formato inválido. Use: treinarmodelo; pergunta; resposta"
        else:
            # Gera a resposta usando o modelo NLP/ML
            return self.nlp_service.generate_answer(user_message)
