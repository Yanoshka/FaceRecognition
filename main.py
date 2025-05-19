# from facenet_pt import FaceRecognizer
#
# FaceRecognizer

# read camera
# try to detect face
# try to recognize, if recognize show name in terminal
#if not make a folder and save photos
#ask name in terminal
import cv2
import torch
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Завантажуємо модель
mtcnn = MTCNN(keep_all=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Завантаження ембеддінгів
def load_embeddings(path='embeddings_template.csv'):
    df = pd.read_csv(path)
    names = df['name'].tolist()
    embeddings = df.drop(columns=['person_id', 'name']).values
    return names, torch.tensor(embeddings, dtype=torch.float32)

# Функція для розпізнавання
def recognize_face(frame, names, embeddings):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face = mtcnn(img)
    if face is not None:
        face_embedding = resnet(face.unsqueeze(0))
        dists = (embeddings - face_embedding).norm(dim=1)
        min_idx = torch.argmin(dists)
        if dists[min_idx] < 0.8:
            return names[min_idx]
    return "Unknown"

# Основна логіка
def main():
    names, embeddings = load_embeddings()
    cap = cv2.VideoCapture(0)

    print("🔍 Запуск розпізнавання обличчя... Натисніть 'q' для виходу.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        name = recognize_face(frame, names, embeddings)
        print(f"🧠 Розпізнано: {name}")

        # Відображення на екрані
        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
