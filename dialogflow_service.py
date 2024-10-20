from google.cloud import dialogflow
import os

class DialogflowService:
    def __init__(self):
        # Coloque aqui o caminho para o arquivo de credenciais do Dialogflow
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./chatbottcc2024-62bc71ec7291.json"
        self.project_id = 'chatbottcc2024'
        self.session_client = dialogflow.SessionsClient()

    def detect_intent_text(self, text, session_id='12345'):
        session = self.session_client.session_path(self.project_id, session_id)
        text_input = dialogflow.TextInput(text=text, language_code="pt-BR")
        query_input = dialogflow.QueryInput(text=text_input)
        response = self.session_client.detect_intent(session=session, query_input=query_input)
        return response.query_result.fulfillment_text