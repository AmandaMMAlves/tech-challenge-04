import cv2
from deepface import DeepFace
import os
import numpy as np
from tqdm import tqdm
import mediapipe as mp

def detect_face_and_activities(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    emotion_color_map = {
        'Feliz': (0, 255, 0),
        'Triste': (255, 0, 0),
        'Raiva': (0, 0, 255),
        'Surpreso': (255, 255, 0),
        'Neutro': (255, 255, 255)
    }

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    for _ in tqdm(range(total_frames), desc="Processando emoções e atividades humanas em vídeo"):
        ret, frame = cap.read()

        if not ret:
            break

        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='retinaface')

        if len(result) == 0:
            continue

        for face in result:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            dominant_emotion = face['dominant_emotion']
            print(dominant_emotion)
            color = emotion_color_map.get(dominant_emotion, (36, 255, 12))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

        # Converter o frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar o frame para detectar a pose
        results = pose.process(rgb_frame)

        print(results)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()



def analyse_video(video_path):
    detect_face_and_activities(video_path, './output_video.mp4')


if __name__ == "__main__":
    analyse_video('./Unlocking Facial Recognition_ Diverse Activities Analysis.mp4') 