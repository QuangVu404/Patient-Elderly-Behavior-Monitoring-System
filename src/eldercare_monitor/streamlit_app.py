from __future__ import annotations

import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import streamlit as st

from eldercare_monitor.config import load_config
from eldercare_monitor.pipeline import MonitorPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_VIDEO = PROJECT_ROOT / "data" / "raw" / "demo_bus.mp4"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
TRAINED_CONFIG = PROJECT_ROOT / "configs" / "trained.yaml"


def _active_config_path() -> Path:
    override = os.environ.get("ELDERCARE_CONFIG")
    if override:
        path = Path(override).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    return TRAINED_CONFIG if TRAINED_CONFIG.exists() else DEFAULT_CONFIG


def _source_value(value: str) -> int | str:
    value = value.strip()
    return int(value) if value.isdigit() else value


def _uploaded_video(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        return Path(handle.name)


def main() -> None:
    st.set_page_config(page_title="Eldercare Monitor", layout="wide")
    st.title("Eldercare Monitor")
    st.caption(
        "Demo phát hiện nguy cơ ngã và nằm bất động bằng pose tracking "
        "và mô hình thời gian đã huấn luyện."
    )

    config_path = _active_config_path()
    with st.sidebar:
        st.header("Nguồn video")
        source_options = ["Tải video", "Camera, RTSP hoặc đường dẫn"]
        if DEMO_VIDEO.exists():
            source_options.insert(0, "Video mẫu")
        source_kind = st.radio("Loại nguồn", source_options)
        uploaded = None
        source_text = "0"
        if source_kind == "Tải video":
            uploaded = st.file_uploader("Chọn video", type=["mp4", "avi", "mov", "mkv"])
        elif source_kind == "Camera, RTSP hoặc đường dẫn":
            source_text = st.text_input("Nguồn", "0", help="Nhập 0 cho webcam hoặc nhập URL RTSP")
        max_frames = st.number_input("Số frame tối đa", 1, 10000, 300)
        start = st.button("Chạy giám sát", type="primary", width="stretch")
        st.caption(f"Cấu hình: {config_path.name}")
        st.info("Dùng nút Stop ở góc trên của Streamlit để dừng sớm.")

    frame_area = st.empty()
    metrics = st.empty()
    event_area = st.empty()

    if not start:
        st.info("Chọn nguồn video rồi nhấn **Chạy giám sát**.")
        return
    if source_kind == "Tải video" and uploaded is None:
        st.warning("Hãy chọn một file video trước.")
        return

    temporary_path: Path | None = None
    source: int | str
    if source_kind == "Video mẫu":
        source = str(DEMO_VIDEO)
    elif uploaded is not None:
        temporary_path = _uploaded_video(uploaded)
        source = str(temporary_path)
    else:
        source = _source_value(source_text)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error(f"Không mở được nguồn video: {source}")
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        return

    try:
        config = load_config(config_path)
        config.video.display = False
        pipeline = MonitorPipeline(config)
    except Exception as exc:
        cap.release()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        st.error(f"Không khởi tạo được pipeline từ {config_path.name}: {exc}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS) or config.sequence.sample_fps
    live_source = isinstance(source, int) or str(source).lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    )
    events: list[dict] = []
    processed = 0
    started_at = time.perf_counter()
    try:
        while processed < int(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            processed += 1
            elapsed = time.perf_counter() - started_at
            fps = processed / max(elapsed, 1e-6)
            media_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_time = (
                time.monotonic()
                if live_source
                else media_time if media_time > 0 else (processed - 1) / source_fps
            )
            annotated, detected = pipeline.process_frame(frame, now=frame_time, fps=fps)
            events.extend(asdict(event) for event in detected)
            frame_area.image(annotated, channels="BGR", width="stretch")
            metrics.metric("Đã xử lý", f"{processed} frame", f"{fps:.1f} FPS")
            if events:
                event_area.dataframe(events, width="stretch")
    finally:
        cap.release()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    st.success(f"Đã xử lý xong {processed} frame, ghi nhận {len(events)} cảnh báo.")


if __name__ == "__main__":
    main()
