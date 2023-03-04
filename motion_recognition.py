import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller

OFFSET = 100
GREEN = (150, 200, 12)
keyboard = Controller()


def resize(image):
    if image.shape[1] > 900:
        coefficient = 900 / image.shape[1]
        width = int(image.shape[1] * coefficient)
        height = int(image.shape[0] * coefficient)
        size = (width, height)
        return cv2.resize(image, size)
    return image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands= 1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        image = resize(image)
        if not success:
            print("Ignoring empty camera frame.")
            continue
        h, w, _ = image.shape
        x_center = w // 2
        y_center = h // 2
        cv2.line(image, (x_center + OFFSET, 0), (x_center + OFFSET, h), GREEN, 2)
        cv2.line(image, (x_center - OFFSET, 0), (x_center - OFFSET, h), GREEN, 2)
        cv2.line(image, (0, y_center + OFFSET), (w, y_center + OFFSET), GREEN, 2)
        cv2.line(image, (0, y_center - OFFSET), (w, y_center - OFFSET), GREEN, 2)
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for id, lm in enumerate(results.multi_hand_landmarks[0].landmark):
                if id == 8:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 20, (255, 0, 255), cv2.FILLED)
                    if cx >= (x_center + OFFSET) and (y_center - OFFSET) <= cy <= (y_center + OFFSET):
                        keyboard.press(Key.left)
                        keyboard.release(Key.left)
                    if cx <= (x_center - OFFSET) and (y_center - OFFSET) <= cy <= y_center + OFFSET:
                        keyboard.press(Key.right)
                        keyboard.release(Key.right)
                    if cy <= (y_center - OFFSET) and (x_center - OFFSET) <= cx <= (x_center + OFFSET):
                        keyboard.press(Key.up)
                        keyboard.release(Key.up)
                    if cy >= (y_center + OFFSET) and (x_center - OFFSET) <= cx <= (x_center + OFFSET):
                        keyboard.press(Key.down)
                        keyboard.release(Key.down)
        cv2.imshow('1', cv2.flip(image, flipCode = 1))
        cv2.waitKey(1)
cap.release()
