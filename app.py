import atexit
import mimetypes
import time
from typing import List

import cv2
import mediapipe as mp
import pyautogui as pgi
from tkinter import *
from tkinter import filedialog, messagebox

from angle_calc import angle_calc


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

POSE_FOR_IMAGES = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
)
POSE_FOR_STREAM = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def _close_pose_solutions() -> None:
    for solution in (POSE_FOR_IMAGES, POSE_FOR_STREAM):
        close = getattr(solution, "close", None)
        if callable(close):
            close()


atexit.register(_close_pose_solutions)

mimetypes.init()

root = Tk()
root.geometry("800x800")
root.title("Biomechanic Posture System")

variable1 = StringVar(value="Rapid Upper Limb Assessment Score : --")
variable2 = StringVar(value="Rapid Entire Body Score : --")

Label(root, text="Biomechanic Posture System", font=("Helvetica", 25, "bold")).place(relx=.5, rely=0, anchor=N)
Label(root, textvariable=variable1, font=("Helvetica", 10, "bold")).place(relx=.5, rely=.6, anchor=N)
Label(root, textvariable=variable2, font=("Helvetica", 10, "bold")).place(relx=.5, rely=.7, anchor=N)


def show_warning(message: str) -> None:
    try:
        pgi.alert(message, "Warning")
    except Exception:
        messagebox.showwarning("Warning", message)


def _extract_landmarks(image, results) -> List[List[float]]:
    pose_points: List[List[float]] = []
    if not results or not results.pose_landmarks:
        return pose_points

    height, width, _ = image.shape
    mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        pose_points.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        cx, cy = int(landmark.x * width), int(landmark.y * height)
        color = (255, 0, 0) if idx % 2 == 0 else (255, 0, 255)
        cv2.circle(image, (cx, cy), 5, color, cv2.FILLED)

    return pose_points


def _update_gui_scores(rula_score: str, reba_score: str) -> None:
    variable1.set(f"Rapid Upper Limb Assessment Score : {rula_score}")
    variable2.set(f"Rapid Entire Body Score : {reba_score}")
    root.update_idletasks()


def _evaluate_scores(pose_landmarks: List[List[float]]) -> tuple[str, str]:
    if not pose_landmarks:
        return "NULL", "NULL"
    return angle_calc(pose_landmarks)


def image_pose_estimation(path: str) -> None:
    img = cv2.imread(path)
    if img is None:
        messagebox.showerror("Error", "Unable to load the selected image.")
        return

    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = POSE_FOR_IMAGES.process(rgb_image)
    pose_landmarks = _extract_landmarks(img, results)

    display_image = cv2.resize(img, (700, 700))
    cv2.imshow("Image", display_image)

    rula_score, reba_score = _evaluate_scores(pose_landmarks)
    _update_gui_scores(rula_score, reba_score)

    if rula_score.isdigit() and int(rula_score) > 3:
        show_warning("Posture not proper in upper body")
    elif reba_score.isdigit() and int(reba_score) > 4:
        show_warning("Posture not proper in your body")
    elif rula_score == "NULL" or reba_score == "NULL":
        show_warning("Posture could not be detected in the selected image.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_pose_estimation(source) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to open the selected video source.")
        return

    frame_index = 0
    try:
        while True:
            frame_index += 20
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = POSE_FOR_STREAM.process(rgb_frame)
            pose_landmarks = _extract_landmarks(frame, results)

            display_frame = cv2.resize(frame, (600, 800))
            cv2.imshow("Image", display_frame)

            rula_score, reba_score = _evaluate_scores(pose_landmarks)
            _update_gui_scores(rula_score, reba_score)

            if rula_score.isdigit() and int(rula_score) > 3:
                show_warning("Posture not proper in upper body")
            elif reba_score.isdigit() and int(reba_score) > 4:
                show_warning("Posture not proper in your body")
            elif rula_score == "NULL" or reba_score == "NULL":
                show_warning("Posture Incorrect")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def webcam() -> None:
    video_pose_estimation(0)


def browsefunc() -> None:
    filename = filedialog.askopenfilename()
    if not filename:
        return

    mime_type = mimetypes.guess_type(filename)[0]
    media_type = mime_type.split('/')[0] if mime_type else None

    if media_type == 'video':
        video_pose_estimation(filename)
    elif media_type == 'image':
        image_pose_estimation(filename)
    else:
        messagebox.showinfo("Unsupported file", "Please choose an image or video file.")


Button(root, text="Browse for a video or an image", font=40, command=browsefunc).place(relx=.5, rely=.2, anchor=N)
Button(root, text="Choose Live Posture Analysis using webcam", font=40, command=webcam).place(relx=.5, rely=.4, anchor=N)

root.mainloop()
