import os
import csv
import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AdamW
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, tokenizer, questions, answers, max_length=128):
        self.tokenizer = tokenizer
        self.questions = questions
        self.answers = answers
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        inputs = self.tokenizer(question, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        labels = self.tokenizer(answer, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt").input_ids
        return {
            'input_ids': inputs['input_ids'].squeeze(), 
            'attention_mask': inputs['attention_mask'].squeeze(), 
            'labels': labels.squeeze()
        }

class NLPService:
    def __init__(self, model_name='facebook/blenderbot-400M-distill'):
        self.device = torch.device("cpu")
        self.fine_tuned_model_path = './models/fine_tuned_model'
        
        # Iniciar com o modelo fine-tuned se ele existir
        if os.path.exists(self.fine_tuned_model_path):
            self.model = BlenderbotForConditionalGeneration.from_pretrained(self.fine_tuned_model_path).to(self.device)
            self.tokenizer = BlenderbotTokenizer.from_pretrained(self.fine_tuned_model_path)
            self.current_model = './models/fine_tuned_model'
            print("Modelo inicial é o fine-tuned.")
        else:
            # Caso não exista, usa o blenderbot como fallback
            self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name).to(self.device)
            self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
            self.current_model = model_name
            print("Modelo inicial é o blenderbot.")

    def generate_answer(self, question):
        csv_answer = self.search_in_csv(question)
        if csv_answer:
            return csv_answer
        
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        output = self.model.generate(
            inputs['input_ids'],
            max_length=150,
            num_return_sequences=1,
            temperature=0.2,
            top_p=0.9,       
            do_sample=True 
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def train_with_input(self, question, answer):
        # Salva a pergunta e resposta no CSV
        with open('data/faq_devs.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow([question, answer])

        # Carrega as perguntas e respostas do CSV
        questions, answers = self.load_data_from_csv('data/faq_devs.csv')

        # Se tiver ao menos 10 entradas, realiza o treinamento
        if len(questions) >= 10:
            self.switch_to_base_model()  # Troca para o modelo base (blenderbot) antes de treinar
            fine_tune_result = self.fine_tune_model('data/faq_devs.csv')
            self.switch_to_fine_tuned_model()  # Volta para o modelo ajustado
            return f"Pergunta e resposta salvas no CSV. {fine_tune_result}"

        return "Pergunta e resposta salvas no CSV. O modelo será treinado posteriormente."

    def fine_tune_model(self, csv_path='data/faq_devs.csv'):
        questions, answers = self.load_data_from_csv(csv_path)

        if len(questions) < 10:
            return "Não há exemplos suficientes no CSV para treinar o modelo. Aguarde mais entradas."

        dataset = CustomDataset(self.tokenizer, questions, answers)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.model.train()

        for epoch in range(10):
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # Salva o modelo ajustado
        if not os.path.exists(self.fine_tuned_model_path):
            os.makedirs(self.fine_tuned_model_path)

        self.model.save_pretrained(self.fine_tuned_model_path)
        self.tokenizer.save_pretrained(self.fine_tuned_model_path)

        return "Modelo ajustado e salvo com sucesso."

    def switch_to_fine_tuned_model(self):
        if os.path.exists(self.fine_tuned_model_path):
            self.model = BlenderbotForConditionalGeneration.from_pretrained(self.fine_tuned_model_path).to(self.device)
            self.tokenizer = BlenderbotTokenizer.from_pretrained(self.fine_tuned_model_path)
            self.current_model = './models/fine_tuned_model'
            print("Modelo trocado para o fine-tuned.")

    def switch_to_base_model(self):
        self.model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill').to(self.device)
        self.tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
        self.current_model = 'facebook/blenderbot-400M-distill'
        print("Modelo trocado para o base (blenderbot).")

    def switch_model(self):
        if self.current_model == 'facebook/blenderbot-400M-distill':
            self.switch_to_fine_tuned_model()
            return "Modelo trocado para o fine-tuned."
        else:
            self.switch_to_base_model()
            return "Modelo trocado para o blenderbot."

    def load_data_from_csv(self, csv_path):
        questions = []
        answers = []
        with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Ignora o cabeçalho
            for row in reader:
                if len(row) < 2:
                    continue
                questions.append(row[0].strip('" '))
                answers.append(row[1].strip('" '))
        return questions, answers

    def search_in_csv(self, question):
        with open('data/faq_devs.csv', mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                if row[0].strip('" ').lower() == question.strip().lower():
                    return row[1].strip('" ')
        return None
