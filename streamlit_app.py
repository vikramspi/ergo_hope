"""Streamlit dashboard for live ergonomics pose analysis."""
from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass
from typing import List, Tuple
import threading

import cv2
import mediapipe as mp
import streamlit as st
import av
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from angle_calc import angle_calc

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


@dataclass(frozen=True)
class PoseStatus:
    rula: str
    reba: str
    message: str
    bgr_colour: Tuple[int, int, int]

    @property
    def hex_colour(self) -> str:
        b, g, r = self.bgr_colour
        return f"#{r:02x}{g:02x}{b:02x}"


def _evaluate_scores(pose_landmarks: List[List[float]]) -> Tuple[str, str]:
    if not pose_landmarks:
        return "NULL", "NULL"
    return angle_calc(pose_landmarks)


def _status_from_scores(rula_score: str, reba_score: str) -> PoseStatus:
    default_colour = (153, 153, 153)
    if rula_score == "NULL" or reba_score == "NULL":
        return PoseStatus(rula_score, reba_score, "Pose not detected", (0, 165, 255))

    try:
        rula_value = int(rula_score)
    except ValueError:
        rula_value = None
    try:
        reba_value = int(reba_score)
    except ValueError:
        reba_value = None

    if rula_value is None or reba_value is None:
        return PoseStatus(rula_score, reba_score, "Awaiting scores", default_colour)

    if rula_value <= 3 and reba_value <= 4:
        return PoseStatus(rula_score, reba_score, "Posture within acceptable range", (0, 180, 0))
    if rula_value > 3 and reba_value > 4:
        return PoseStatus(rula_score, reba_score, "Whole body posture requires attention", (0, 0, 255))
    if rula_value > 3:
        return PoseStatus(rula_score, reba_score, "Upper body posture requires attention", (0, 0, 255))
    if reba_value > 4:
        return PoseStatus(rula_score, reba_score, "Lower body posture requires attention", (0, 0, 255))

    return PoseStatus(rula_score, reba_score, "Review posture", default_colour)


class PoseVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lock = threading.Lock()
        self._latest_status: PoseStatus = PoseStatus("--", "--", "Waiting for webcam", (120, 120, 120))

    def _extract_landmarks(self, image, results) -> List[List[float]]:
        pose_points: List[List[float]] = []
        if not results or not results.pose_landmarks:
            return pose_points

        height, width, _ = image.shape
        mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            pose_points.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            colour = (255, 0, 0) if idx % 2 == 0 else (255, 0, 255)
            cv2.circle(image, (cx, cy), 4, colour, cv2.FILLED)
        return pose_points

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb_frame)
        pose_landmarks = self._extract_landmarks(image, results)

        rula_score, reba_score = _evaluate_scores(pose_landmarks)
        status = _status_from_scores(rula_score, reba_score)

        overlay_text = f"RULA: {status.rula} | REBA: {status.reba}"
        cv2.putText(
            image,
            overlay_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            status.message,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status.bgr_colour,
            2,
            cv2.LINE_AA,
        )

        with self._lock:
            self._latest_status = status

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def get_latest_status(self) -> PoseStatus:
        with self._lock:
            return self._latest_status

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        close = getattr(self._pose, "close", None)
        if callable(close):
            close()


def _render_metrics(column, status: PoseStatus) -> None:
    column.metric("RULA Score", status.rula)
    column.metric("REBA Score", status.reba)
    column.markdown(
        f"<div style='padding: 1rem; border-radius: 0.5rem; background-color: {status.hex_colour}20; color: {status.hex_colour}; font-size: 1.2rem; font-weight: 600;'>"
        f"{status.message}"
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Ergonomics Pose Analysis Dashboard", layout="wide")
    st.title("Ergonomics Pose Analysis Dashboard")
    st.write("Use the toggle below to start or stop the live webcam posture analysis.")

    run_stream = st.checkbox("Start live webcam analysis", value=False)
    video_column, score_column = st.columns([3, 2])

    status = PoseStatus("--", "--", "Webcam idle", (153, 153, 153))
    ctx = None
    if run_stream:
        ctx = webrtc_streamer(
            key="pose-analysis",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=PoseVideoProcessor,
            async_processing=True,
        )
        if ctx and ctx.video_processor:
            status = ctx.video_processor.get_latest_status()
        else:
            video_column.info("Connecting to webcam...")
    else:
        video_column.info("Enable the checkbox to start streaming from your webcam.")

    _render_metrics(score_column, status)

    st.markdown(
        "---\n"
        "**Tip:** If the video feed does not appear, make sure your browser has permission to access the webcam."
    )


if __name__ == "__main__":
    main()
