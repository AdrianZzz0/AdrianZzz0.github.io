import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QVBoxLayout, QLabel, QLineEdit, QPushButton, QWidget, QProgressBar, QListWidget
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt
import tensorflow as tf
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Preprocesar texto
def preprocess_text(text):
    print(f"Texto original: {text}")  # Para ver cómo se recibe
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    print(f"Texto procesado: {text}")  # Para ver el resultado
    return text

# Cargar el dataset y preprocesarlo
with open('dataset.json', 'r') as file:
    data = json.load(file)

questions = [preprocess_text(item['question']) for item in data]
answers = [item['answer'] for item in data]

# Vectorizar las preguntas con TF-IDF
tfidf_vectorizer = TfidfVectorizer().fit(questions)
questions_tfidf = tfidf_vectorizer.transform(questions)

# Función para obtener respuesta usando similitud de coseno
def get_response(question):
    question = preprocess_text(question)
    user_tfidf = tfidf_vectorizer.transform([question])  # Vectorizar la pregunta del usuario

    # Calcular similitud coseno con las preguntas del dataset
    similarities = cosine_similarity(user_tfidf, questions_tfidf)
    max_similarity_index = np.argmax(similarities)  # Índice de la pregunta más similar
    max_similarity = similarities[0, max_similarity_index]

    print(f"Similitud máxima: {max_similarity}")  # Depuración

    # Umbral de similitud para aceptar una respuesta
    if max_similarity < 0.5:
        return "No estoy seguro de cómo responder eso."
    
    # Devolver la respuesta correspondiente
    return answers[max_similarity_index]

# Interfaz gráfica
class ChatbotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chatbot con Estilo Personalizado")
        self.resize(800, 600)

        # Configurar fondo
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setBrush(QPalette.Window, QColor(0, 0, 0))  # Fondo negro
        self.setPalette(palette)

        self.layout = QVBoxLayout()

        # Título
        self.title_label = QLabel("Chatbot Personalizado")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        self.layout.addWidget(self.title_label)

        # Lista de preguntas posibles
        self.questions_label = QLabel("Preguntas que puedes hacer:")
        self.questions_label.setStyleSheet("color: white; font-size: 16px; margin-top: 20px;")
        self.layout.addWidget(self.questions_label)

        self.questions_list = QListWidget()
        self.questions_list.addItems(questions)  # Añadimos las preguntas del dataset
        self.questions_list.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.7);
            color: #ffffff;
            border-radius: 10px;
            padding: 10px;
            font-size: 1rem;
            margin-bottom: 20px;
        """)
        self.layout.addWidget(self.questions_list)

        # Entrada de pregunta
        self.question_label = QLabel("Escribe tu pregunta:")
        self.question_label.setStyleSheet("color: white; font-size: 16px;")
        self.layout.addWidget(self.question_label)

        self.question_input = QLineEdit()
        self.question_input.setStyleSheet("padding: 10px; border: 2px solid #ccc; border-radius: 10px; font-size: 14px;")
        self.layout.addWidget(self.question_input)

        # Botón de preguntar
        self.submit_button = QPushButton("Preguntar")
        self.submit_button.setStyleSheet("""
            background-color: #ffffff;
            color: #000000;
            border-radius: 10px;
            padding: 10px;
            font-size: 14px;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        """)
        self.submit_button.clicked.connect(self.handle_question)
        self.layout.addWidget(self.submit_button)

        # Respuesta
        self.answer_label = QLabel("Respuesta:")
        self.answer_label.setStyleSheet("color: white; font-size: 16px; margin-top: 20px;")
        self.layout.addWidget(self.answer_label)

        # Barra de progreso
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #76c7c0;
                width: 20px;
            }
        """)
        self.layout.addWidget(self.progress_bar)

        # Botón de salir
        self.exit_button = QPushButton("Salir")
        self.exit_button.setStyleSheet("""
            background-color: #ff4c4c;
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-size: 14px;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        """)
        self.exit_button.clicked.connect(self.close)
        self.layout.addWidget(self.exit_button)

        self.setLayout(self.layout)

    def handle_question(self):
        question = self.question_input.text()
        self.progress_bar.setValue(50)
        response = get_response(question)
        self.progress_bar.setValue(100)
        self.answer_label.setText(f"Respuesta: {response}")

# Ejecutar la aplicación
app = QApplication(sys.argv)
chatbot_app = ChatbotApp()
chatbot_app.show()
sys.exit(app.exec())
