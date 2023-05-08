# importação bibliotecas utilizadas no projeto
import pandas as pd
import numpy as np 
import cv2
import os
import threading
from PIL import Image

def resize_image(input_path, output_path, size):
    # Abre a imagem e redimensiona
    with Image.open(input_path) as image:
        new_image = image.resize(size)
        # Salva a imagem redimensionada
        new_image.save(output_path)

def resize_images_in_directory(directory_path, output_directory_path, size, num_threads):
    # Cria um objeto Thread para cada imagem a ser redimensionada
    threads = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg"):
            input_path = os.path.join(directory_path, filename)
            output_path = os.path.join(output_directory_path, filename)
            thread = threading.Thread(target=resize_image, args=(input_path, output_path, size))
            threads.append(thread)

    # Inicia todas as threads
    for thread in threads:
        thread.start()

    # Aguarda todas as threads terminarem
    for thread in threads:
        thread.join()

def detect_faces(image_path, output_path, face_cascade):
    # Carrega a imagem e converte para escala de cinza
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecta rostos na imagem
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Desenha retângulos em torno dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Salva a imagem com os rostos detectados
    cv2.imwrite(output_path, img)

def detect_faces_in_directory(directory_path, output_directory_path, face_cascade_path, num_threads):
    # Carrega o classificador Haar Cascade para detecção de rostos
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Cria um objeto Thread para cada imagem a ser processada
    threads = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg"):
            input_path = os.path.join(directory_path, filename)
            output_path = os.path.join(output_directory_path, filename)
            thread = threading.Thread(target=detect_faces, args=(input_path, output_path, face_cascade))
            threads.append(thread)

    # Inicia todas as threads
    for thread in threads:
        thread.start()

    # Aguarda todas as threads terminarem
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    # Define o diretório de entrada e saída das imagens
    input_directory = 'valid'
    output_directory = 'exit'
    
    # Define o tamanho de redimensionamento
    size = (250, 250)

    # Define o número de threads a serem usadas
    num_threads = 4

    # Redimensiona as imagens usando threads
    resize_images_in_directory(input_directory, output_directory, size, num_threads)

    # Define o caminho para o classificador Haar Cascade
    face_cascade_path = 'haarcascade_frontalface_default.xml'

    # Detecta rostos nas imagens usando threads
    detect_faces_in_directory(input_directory, output_directory, face_cascade_path, num_threads)
