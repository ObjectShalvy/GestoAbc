import cv2
import mediapipe as mp
import math
import numpy as np

# Inicializar MediaPipe Hands con mayor precisión
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def calculate_angle(point1, point2, point3):
    radians = math.atan2(point3.y - point2.y, point3.x - point2.x) - \
              math.atan2(point1.y - point2.y, point1.x - point2.x)
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def is_finger_extended(landmarks, tip_idx, mid_idx, base_idx):
    return (landmarks[tip_idx].y < landmarks[mid_idx].y and 
            landmarks[mid_idx].y < landmarks[base_idx].y)

def detect_letter(landmarks):
    """Detecta letras basadas en el alfabeto dactilológico universal"""
    
    # Índices de los puntos clave de los dedos
    thumb_tip = 4
    index_tip = 8
    middle_tip = 12
    ring_tip = 16
    pinky_tip = 20

    # Verificar estado de cada dedo
    thumb_extended = landmarks[thumb_tip].x > landmarks[thumb_tip-1].x
    index_extended = is_finger_extended(landmarks, index_tip, index_tip-1, index_tip-2)
    middle_extended = is_finger_extended(landmarks, middle_tip, middle_tip-1, middle_tip-2)
    ring_extended = is_finger_extended(landmarks, ring_tip, ring_tip-1, ring_tip-2)
    pinky_extended = is_finger_extended(landmarks, pinky_tip, pinky_tip-1, pinky_tip-2)

    # Detección de letras específicas basadas en la imagen de referencia
    
    # A - Puño cerrado con pulgar hacia arriba
    if (thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended])):
        return "E"
        
    
    # B - Dedos extendidos juntos, pulgar cruzado
    elif (all([index_extended, middle_extended, ring_extended, pinky_extended]) and 
          not thumb_extended and 
          calculate_distance(landmarks[index_tip], landmarks[pinky_tip]) < 0.1):
        return "B"
    
    # C - Dedos juntos curvados
    elif (not any([index_extended, middle_extended, ring_extended, pinky_extended]) and
          calculate_distance(landmarks[thumb_tip], landmarks[index_tip]) > 0.1):
        return "C"
    
    # D - Índice extendido, pulgar tocando dedos
    elif (index_extended and not any([middle_extended, ring_extended, pinky_extended])):
        return "D"
    
    # E - Dedos doblados, pulgar cubierto
    elif not any([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
        return "A"
    
    # F - Índice y pulgar unidos, otros dedos extendidos
    elif (calculate_distance(landmarks[thumb_tip], landmarks[index_tip]) < 0.05 and
          all([middle_extended, ring_extended, pinky_extended])):
        return "F"
    
    # G - Índice apuntando al pulgar
    elif (index_extended and not any([middle_extended, ring_extended, pinky_extended]) and
          landmarks[index_tip].x < landmarks[thumb_tip].x):
        return "G"
    
    # H - Índice y medio extendidos horizontalmente
    elif (index_extended and middle_extended and
          not any([ring_extended, pinky_extended]) and
          abs(landmarks[index_tip].y - landmarks[middle_tip].y) < 0.05):
        return "H"
    
    # I - Meñique extendido
    elif (pinky_extended and not any([thumb_extended, index_extended, middle_extended, ring_extended])):
        return "I"
    
    # J - Meñique extendido con movimiento (simplificado)
    elif (pinky_extended and not any([index_extended, middle_extended, ring_extended])):
        return "Y"
    
    # K - Índice y medio extendidos en V
    elif (index_extended and middle_extended and
          not any([ring_extended, pinky_extended]) and
          calculate_distance(landmarks[index_tip], landmarks[middle_tip]) > 0.1):
        return "K"
    
    # L - Índice y pulgar en L
    elif (index_extended and thumb_extended and
          not any([middle_extended, ring_extended, pinky_extended])):
        return "L"
    
    # M - Tres dedos sobre el pulgar
    elif (all([index_extended, middle_extended, ring_extended]) and
          not pinky_extended):
        return "M"
    
    # N - Dos dedos sobre el pulgar
    elif (index_extended and middle_extended and
          not any([ring_extended, pinky_extended])):
        return "N"
    
    # O - Dedos formando O
    elif (calculate_distance(landmarks[thumb_tip], landmarks[index_tip]) < 0.05 and
          not any([middle_extended, ring_extended, pinky_extended])):
        return "O"
    
    # P - Índice apuntando hacia abajo
    elif (index_extended and landmarks[index_tip].y > landmarks[index_tip-1].y):
        return "P"
    
    # Q - Similar a G pero más abajo
    elif (index_extended and not any([middle_extended, ring_extended, pinky_extended]) and
          landmarks[index_tip].y > landmarks[0].y):
        return "Q"
    
    # R - Dedos cruzados
    elif (index_extended and middle_extended and
          calculate_distance(landmarks[index_tip], landmarks[middle_tip]) < 0.05):
        return "R"
    
    # S - Puño cerrado con pulgar sobre dedos
    elif not any([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
        return "S"
    
    # T - Índice doblado sobre pulgar
    elif (not index_extended and not any([middle_extended, ring_extended, pinky_extended]) and
          calculate_distance(landmarks[thumb_tip], landmarks[index_tip]) < 0.05):
        return "T"
    
    # U - Índice y medio extendidos juntos
    elif (index_extended and middle_extended and
          not any([ring_extended, pinky_extended]) and
          calculate_distance(landmarks[index_tip], landmarks[middle_tip]) < 0.05):
        return "U"
    
    # V - Índice y medio extendidos en V
    elif (index_extended and middle_extended and
          not any([ring_extended, pinky_extended]) and
          calculate_distance(landmarks[index_tip], landmarks[middle_tip]) > 0.1):
        return "V"
    
    # W - Índice, medio y anular extendidos
    elif (index_extended and middle_extended and ring_extended and
          not pinky_extended):
        return "W"
    
    # X - Índice doblado
    elif (not index_extended and not any([middle_extended, ring_extended, pinky_extended]) and
          landmarks[index_tip].y > landmarks[index_tip-1].y):
        return "X"
    
    # Y - Meñique y pulgar extendidos
    elif (pinky_extended and thumb_extended and
          not any([index_extended, middle_extended, ring_extended])):
        return "Y"
    
    # Z - Índice haciendo zigzag (simplificado)
    elif (index_extended and not any([middle_extended, ring_extended, pinky_extended]) and
          landmarks[index_tip].x < landmarks[index_tip-1].x):
        return "Z"
    
    return "?"

def main():
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar landmarks
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Detectar y mostrar letra
                letter = detect_letter(hand_landmarks.landmark)
                cv2.putText(
                    frame,
                    f'Letra: {letter}',
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA
                )
        
        cv2.imshow('Lenguaje de Señas', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()