from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

try:
    import pyvirtualcam
except ImportError:
    pyvirtualcam = None


MODEL_URL = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx"
MODEL_NAME = "inswapper_128.onnx"
WINDOW_NAME = "LocalFaceSwap Python"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
UPLOADS_POLL_SECONDS = 0.75

PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}


def draw_outlined_text(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.6) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def lower_suffix(path: Path) -> str:
    return path.suffix.lower()


def is_supported_image(path: Path) -> bool:
    return path.is_file() and lower_suffix(path) in SUPPORTED_EXTENSIONS


def find_latest_upload_image(uploads_dir: Path) -> Optional[Path]:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    candidates = [entry for entry in uploads_dir.iterdir() if is_supported_image(entry)]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.name))


def best_face(faces: list[Any]) -> Optional[Any]:
    if not faces:
        return None
    return max(
        faces,
        key=lambda face: (
            float(face.det_score),
            float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
        ),
    )


def ensure_model_file(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return model_path

    print(f"[python-swap] Downloading {MODEL_NAME} to {model_path} ...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print(f"[python-swap] Download complete: {model_path}")
    return model_path


def resolve_execution_providers(preferred: str) -> tuple[list[str], str]:
    available = onnxruntime.get_available_providers()
    if not available:
        raise RuntimeError("No ONNX Runtime execution providers are available.")

    if preferred == "auto":
        for actual in (
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CoreMLExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ):
            if actual in available:
                label = next((alias for alias, name in PROVIDER_ALIASES.items() if name == actual), actual)
                providers = [actual]
                if actual != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
                    providers.append("CPUExecutionProvider")
                return providers, label
        return [available[0]], available[0]

    actual = PROVIDER_ALIASES.get(preferred.lower())
    if actual and actual in available:
        providers = [actual]
        if actual != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        return providers, preferred.lower()

    print(
        f"[python-swap] Requested execution provider '{preferred}' is not available. "
        f"Falling back to CPU. Available: {', '.join(available)}"
    )
    return ["CPUExecutionProvider"], "cpu"


class FrameGrabber:
    def __init__(self, camera_index: int, backend: str, width: int, height: int, fps: int) -> None:
        self.camera_index = camera_index
        self.backend = backend
        self.width = width
        self.height = height
        self.fps = fps
        self.capture: Optional[cv2.VideoCapture] = None
        self.running = False
        self.worker: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.latest: Optional[np.ndarray] = None
        self.frame_id = 0

    def open(self) -> bool:
        backends: list[int] = []
        if self.backend == "dshow" and hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
        backends.append(cv2.CAP_ANY)

        for backend in backends:
            capture = cv2.VideoCapture(self.camera_index, backend)
            if capture.isOpened():
                self.capture = capture
                break
            capture.release()

        if self.capture is None or not self.capture.isOpened():
            return False

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def start(self) -> None:
        self.running = True
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.running = False
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=1.0)
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def latest_frame(self) -> tuple[Optional[np.ndarray], int]:
        with self.lock:
            if self.latest is None:
                return None, self.frame_id
            return self.latest.copy(), self.frame_id

    def _run(self) -> None:
        while self.running and self.capture is not None:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest = frame
                self.frame_id += 1


class InsightFaceEngine:
    def __init__(self, repo_root: Path, providers: list[str], provider_label: str, det_size: int) -> None:
        self.repo_root = repo_root
        self.providers = providers
        self.provider_label = provider_label
        self.det_size = det_size
        self.model_path = ensure_model_file(repo_root / "models" / MODEL_NAME)
        self.ctx_id = -1 if providers == ["CPUExecutionProvider"] else 0
        self.face_analyser = FaceAnalysis(
            name="buffalo_l",
            root=str(repo_root),
            providers=providers,
            allowed_modules=["detection", "recognition", "landmark_2d_106"],
        )
        self.face_analyser.prepare(ctx_id=self.ctx_id, det_size=(det_size, det_size))
        self.face_swapper = get_model(str(self.model_path), providers=providers)

    def extract_source_face(self, image_path: Path) -> tuple[Optional[Any], Optional[np.ndarray], str]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None, None, f"Failed to read image: {image_path.name}"

        faces = self.face_analyser.get(image)
        source_face = best_face(faces)
        if source_face is None:
            return None, image, f"No face found in {image_path.name}"

        return source_face, image, f"Loaded source face from {image_path.name}"

    def swap_frame(self, frame: np.ndarray, source_face: Any, swap_all_faces: bool, opacity: float) -> tuple[np.ndarray, int]:
        faces = self.face_analyser.get(frame)
        if not faces:
            return frame, 0

        targets = faces if swap_all_faces else [best_face(faces)]
        result = frame
        swapped_count = 0
        for target_face in targets:
            if target_face is None:
                continue
            result = self.face_swapper.get(result, target_face, source_face, paste_back=True)
            swapped_count += 1

        if opacity >= 0.999:
            return result, swapped_count

        blended = cv2.addWeighted(frame, 1.0 - opacity, result, opacity, 0.0)
        return blended, swapped_count


class VirtualCameraOutput:
    def __init__(self, fps: int, backend: str, device: Optional[str]) -> None:
        self.fps = fps
        self.backend = backend
        self.device = device
        self.camera: Optional[Any] = None
        self.device_name = ""

    def send(self, frame: np.ndarray) -> None:
        if self.camera is None:
            self._open(frame)
        assert self.camera is not None
        self.camera.send(frame)

    def close(self) -> None:
        if self.camera is not None:
            self.camera.close()
            self.camera = None

    def _open(self, frame: np.ndarray) -> None:
        if pyvirtualcam is None:
            raise RuntimeError(
                "pyvirtualcam is not installed. Re-run setup-python.ps1 to install virtual camera support."
            )

        backend = None if self.backend == "auto" else self.backend
        try:
            self.camera = pyvirtualcam.Camera(
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                fps=float(self.fps),
                fmt=pyvirtualcam.PixelFormat.BGR,
                device=self.device,
                backend=backend,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to open a virtual camera. Install OBS Studio or Unity Capture first, then select the "
                f"virtual camera in Google Meet. Backend={self.backend} Device={self.device or 'auto'} Error: {exc}"
            ) from exc

        self.device_name = self.camera.device
        print(f"[python-swap] Virtual camera ready: {self.device_name}")


@dataclass
class SourceState:
    path: Optional[Path] = None
    modified_ns: int = 0
    face: Optional[Any] = None
    image: Optional[np.ndarray] = None
    status: str = "Drop a JPG/PNG into uploads/ to start."


class LiveFaceSwapApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_root = Path(__file__).resolve().parents[1]
        self.uploads_dir = (self.repo_root / args.uploads_dir).resolve() if not Path(args.uploads_dir).is_absolute() else Path(args.uploads_dir)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.providers, self.provider_label = resolve_execution_providers(args.execution_provider)
        self.engine = InsightFaceEngine(self.repo_root, self.providers, self.provider_label, args.det_size)
        self.grabber = FrameGrabber(args.camera, args.backend, args.width, args.height, args.fps)
        self.source_state = SourceState()
        self.last_upload_poll = 0.0
        self.last_message = ""
        self.last_processed_frame_id = -1
        self.fps_estimate = 0.0
        self.previous_frame_time = time.perf_counter()
        self.virtual_camera = (
            VirtualCameraOutput(args.fps, args.virtual_camera_backend, args.virtual_camera_device)
            if args.virtual_camera
            else None
        )

    def refresh_source_image(self, force: bool = False) -> None:
        latest = find_latest_upload_image(self.uploads_dir)
        if latest is None:
            if force or self.source_state.path is not None:
                self.source_state = SourceState(status="Drop a JPG/PNG into uploads/ to start.")
                self._print_once("[python-swap] Waiting for a JPG/PNG in uploads/.")
            return

        modified_ns = latest.stat().st_mtime_ns
        if not force and self.source_state.path == latest and self.source_state.modified_ns == modified_ns:
            return

        source_face, image, status = self.engine.extract_source_face(latest)
        self.source_state.path = latest
        self.source_state.modified_ns = modified_ns
        self.source_state.face = source_face
        self.source_state.image = image
        self.source_state.status = status
        prefix = "[python-swap] "
        if source_face is None:
            self._print_once(prefix + status)
        else:
            self._print_once(prefix + status)

    def run(self) -> int:
        self.refresh_source_image(force=True)
        if not self.grabber.open():
            print("[python-swap] Failed to open camera.")
            return 1

        print(f"[python-swap] Watching uploads folder: {self.uploads_dir}")
        print(f"[python-swap] Execution provider: {', '.join(self.providers)}")
        if self.args.preview:
            print("[python-swap] Keys: q quit | r rescan uploads | m preview mirror toggle")
        else:
            print("[python-swap] Running without a local preview window. Press Ctrl+C to stop.")

        self.grabber.start()
        try:
            while True:
                now = time.perf_counter()
                if now - self.last_upload_poll >= UPLOADS_POLL_SECONDS:
                    self.last_upload_poll = now
                    self.refresh_source_image(force=False)

                frame, frame_id = self.grabber.latest_frame()
                if frame is None or frame_id == self.last_processed_frame_id:
                    time.sleep(0.005)
                    continue
                self.last_processed_frame_id = frame_id

                output_frame = frame
                swapped_count = 0
                if self.source_state.face is not None:
                    try:
                        output_frame, swapped_count = self.engine.swap_frame(
                            output_frame,
                            self.source_state.face,
                            self.args.swap_all_faces,
                            self.args.opacity,
                        )
                    except Exception as exc:
                        self.source_state.status = f"Swap failed: {exc}"
                else:
                    self.source_state.status = self.source_state.status or "Drop a JPG/PNG into uploads/ to start."

                if self.virtual_camera is not None:
                    try:
                        self.virtual_camera.send(output_frame)
                    except Exception as exc:
                        print(f"[python-swap] Virtual camera output failed: {exc}")
                        return 1

                current_time = time.perf_counter()
                seconds = current_time - self.previous_frame_time
                self.previous_frame_time = current_time
                if seconds > 0:
                    instant_fps = 1.0 / seconds
                    self.fps_estimate = instant_fps if self.fps_estimate <= 0 else (self.fps_estimate * 0.90) + (instant_fps * 0.10)

                if self.args.preview:
                    display = cv2.flip(output_frame, 1) if self.args.mirror else output_frame
                    self.draw_hud(display, swapped_count)
                    cv2.imshow(WINDOW_NAME, display)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("r"):
                        self.refresh_source_image(force=True)
                    if key == ord("m"):
                        self.args.mirror = not self.args.mirror
        finally:
            self.grabber.stop()
            if self.virtual_camera is not None:
                self.virtual_camera.close()
            cv2.destroyAllWindows()
        return 0

    def draw_hud(self, frame: np.ndarray, swapped_count: int) -> None:
        draw_outlined_text(frame, "python-live-swap", (16, 28), 0.7)

        stats = f"fps {int(round(self.fps_estimate))}  provider {self.provider_label}"
        if self.source_state.path is not None:
            stats += f"  img {self.source_state.path.name}"
        if self.virtual_camera is not None:
            stats += "  vcam on"
        draw_outlined_text(frame, stats, (16, 54), 0.55)

        mode = "all-faces" if self.args.swap_all_faces else "single-face"
        tune = (
            f"mode {mode}  opacity {int(round(self.args.opacity * 100))}%"
            f"  preview {'on' if self.args.preview else 'off'}"
            f"  mirror {'on' if self.args.mirror else 'off'}"
        )
        draw_outlined_text(frame, tune, (16, 80), 0.5)

        if self.source_state.face is None:
            draw_outlined_text(frame, self.source_state.status, (16, frame.shape[0] - 24), 0.55)
        elif swapped_count == 0:
            draw_outlined_text(frame, "No live face detected yet.", (16, frame.shape[0] - 24), 0.55)

    def _print_once(self, message: str) -> None:
        if message != self.last_message:
            self.last_message = message
            print(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local folder-based live face swap with InsightFace.")
    parser.add_argument("--uploads-dir", default="uploads", help="Folder watched for JPG/PNG source images.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS.")
    parser.add_argument("--det-size", type=int, default=640, help="Face detector input size.")
    parser.add_argument(
        "--execution-provider",
        default="auto",
        choices=["auto", "cpu", "cuda", "directml", "openvino", "coreml", "tensorrt"],
        help="ONNX Runtime execution provider preference.",
    )
    parser.add_argument("--backend", default="dshow" if os.name == "nt" else "any", choices=["dshow", "any"], help="Camera backend.")
    parser.add_argument("--mirror", dest="mirror", action="store_true", default=True, help="Mirror the local preview window.")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false", help="Disable local preview mirroring.")
    parser.add_argument("--preview", dest="preview", action="store_true", default=True, help="Show the local preview window.")
    parser.add_argument("--no-preview", dest="preview", action="store_false", help="Run in the background without a local preview window.")
    parser.add_argument("--virtual-camera", action="store_true", help="Publish the swapped output to a virtual camera for apps like Google Meet.")
    parser.add_argument(
        "--virtual-camera-backend",
        default="auto",
        choices=["auto", "obs", "unitycapture"],
        help="Preferred Windows virtual camera backend.",
    )
    parser.add_argument("--virtual-camera-device", default=None, help="Exact virtual camera device name, if you want to force one.")
    parser.add_argument("--swap-all-faces", action="store_true", help="Swap every detected face instead of only the main face.")
    parser.add_argument("--opacity", type=float, default=1.0, help="Blend swapped face with original frame, from 0.0 to 1.0.")
    args = parser.parse_args()
    args.opacity = max(0.0, min(1.0, args.opacity))
    if not args.preview and not args.virtual_camera:
        parser.error("--no-preview only makes sense together with --virtual-camera.")
    return args


def main() -> int:
    args = parse_args()
    try:
        app = LiveFaceSwapApp(args)
    except Exception as exc:
        print(f"[python-swap] Startup failed: {exc}")
        print("[python-swap] Make sure the Python requirements are installed and the first model download is allowed.")
        return 1
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
