"""
QueryMind face recognition using OpenCV YuNet + SFace.

This is custom biometric authentication, not a Clerk passkey/WebAuthn credential.
The face embedding is stored in Supabase and is used only for matching.
"""
from __future__ import annotations

import base64
import binascii
import os
from typing import Any

import cv2
import numpy as np

from backend.services.auth import get_face_embeddings_for_matching, save_face_embedding, has_face_registered, get_user_by_email

from backend.core.config import BACKEND_DIR

_BACKEND_DIR = str(BACKEND_DIR)
_MODELS_DIR = os.path.join(_BACKEND_DIR, "models")
_DETECTOR_MODEL = os.path.join(_MODELS_DIR, "face_detection_yunet_2023mar.onnx")
_RECOGNIZER_MODEL = os.path.join(_MODELS_DIR, "face_recognition_sface_2021dec.onnx")

FACE_SCORE_THRESHOLD = float(os.getenv("FACE_DETECTION_SCORE_THRESHOLD", "0.80"))
FACE_COSINE_THRESHOLD = float(os.getenv("FACE_COSINE_THRESHOLD", "0.45"))
MAX_IMAGE_BYTES = 5 * 1024 * 1024

_detector = None
_recognizer = None


def _load_models():
    global _detector, _recognizer
    if _detector is None:
        if not os.path.exists(_DETECTOR_MODEL):
            raise RuntimeError(f"YuNet model not found: {_DETECTOR_MODEL}")
        # Initialize detector with 0.60 so YuNet outputs candidate detections for our post-processing
        _detector = cv2.FaceDetectorYN.create(
            _DETECTOR_MODEL, "", (320, 320), min(0.60, FACE_SCORE_THRESHOLD), 0.3, 5000
        )
    if _recognizer is None:
        if not os.path.exists(_RECOGNIZER_MODEL):
            raise RuntimeError(f"SFace model not found: {_RECOGNIZER_MODEL}")
        _recognizer = cv2.FaceRecognizerSF.create(_RECOGNIZER_MODEL, "")
    return _detector, _recognizer


def calculate_iou(boxA: list[float], boxB: list[float]) -> tuple[float, float]:
    """Calculate IoU and containment ratio between two bounding boxes [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0.0, boxA[2]) * max(0.0, boxA[3])
    boxBArea = max(0.0, boxB[2]) * max(0.0, boxB[3])

    unionArea = boxAArea + boxBArea - interArea
    iou = (interArea / unionArea) if unionArea > 0 else 0.0

    minArea = min(boxAArea, boxBArea)
    containment = (interArea / minArea) if minArea > 0 else 0.0

    return iou, containment


def filter_faces(
    faces: np.ndarray | list | None,
    width: int,
    height: int,
    score_threshold: float = FACE_SCORE_THRESHOLD,
    min_size: float = 40.0,
    min_area_ratio: float = 0.010,
    iou_threshold: float = 0.35,
    containment_threshold: float = 0.60,
) -> list[np.ndarray]:
    """
    Robust post-processing for YuNet face detections:
    1. Logs raw detections and bounding boxes.
    2. Filters low-confidence detections.
    3. Filters tiny/small false positive boxes.
    4. Deduplicates overlapping multi-scale/sub-box detections on the same face.
    """
    if faces is None or len(faces) == 0:
        print("[face] raw_detections=0", flush=True)
        print("[face] valid_detections=0", flush=True)
        print("[face] rejected_small=0", flush=True)
        print("[face] rejected_low_confidence=0", flush=True)
        print("[face] rejected_duplicate=0", flush=True)
        return []

    raw_count = len(faces)
    print(f"[face] raw_detections={raw_count}", flush=True)

    for i, face in enumerate(faces):
        x, y, w, h = [float(v) for v in face[:4]]
        conf = float(face[-1])
        print(
            f"[face] box: confidence={conf:.4f}, x={x:.1f}, y={y:.1f}, width={w:.1f}, height={h:.1f}",
            flush=True,
        )

    # 1. Filter low-confidence detections
    valid_after_conf = []
    rejected_low_conf = 0
    for face in faces:
        conf = float(face[-1])
        if conf >= score_threshold:
            valid_after_conf.append(face)
        else:
            rejected_low_conf += 1
            x, y, w, h = [float(v) for v in face[:4]]
            print(
                f"[face] rejected_low_confidence: confidence={conf:.4f}, x={x:.1f}, y={y:.1f}, width={w:.1f}, height={h:.1f}",
                flush=True,
            )

    # 2. Filter tiny / small boxes relative to frame dimensions
    valid_after_size = []
    rejected_small = 0
    frame_area = float(width * height)

    for face in valid_after_conf:
        x, y, w, h = [float(v) for v in face[:4]]
        area = max(0.0, w) * max(0.0, h)
        area_ratio = area / frame_area if frame_area > 0 else 0.0
        conf = float(face[-1])

        if w < min_size or h < min_size or area_ratio < min_area_ratio:
            rejected_small += 1
            print(
                f"[face] rejected_small: confidence={conf:.4f}, x={x:.1f}, y={y:.1f}, width={w:.1f}, height={h:.1f}",
                flush=True,
            )
        else:
            valid_after_size.append(face)

    # 3. Sort candidates by confidence descending
    valid_after_size.sort(key=lambda f: float(f[-1]), reverse=True)

    # 4. Overlap / IoU deduplication (Non-Maximum Suppression)
    final_faces = []
    rejected_duplicate = 0

    for face in valid_after_size:
        box = [float(v) for v in face[:4]]
        conf = float(face[-1])
        is_dup = False
        for kept_face in final_faces:
            kept_box = [float(v) for v in kept_face[:4]]
            iou, containment = calculate_iou(box, kept_box)
            if iou >= iou_threshold or containment >= containment_threshold:
                is_dup = True
                rejected_duplicate += 1
                print(
                    f"[face] rejected_duplicate: confidence={conf:.4f}, x={box[0]:.1f}, y={box[1]:.1f}, width={box[2]:.1f}, height={box[3]:.1f}",
                    flush=True,
                )
                break
        if not is_dup:
            final_faces.append(face)

    valid_count = len(final_faces)
    print(f"[face] valid_detections={valid_count}", flush=True)
    print(f"[face] rejected_small={rejected_small}", flush=True)
    print(f"[face] rejected_low_confidence={rejected_low_conf}", flush=True)
    print(f"[face] rejected_duplicate={rejected_duplicate}", flush=True)

    return final_faces


def decode_image_data(image_data: str) -> np.ndarray:
    if not isinstance(image_data, str) or not image_data:
        raise ValueError("Image data is required.")

    payload = image_data
    if "," in payload:
        header, payload = payload.split(",", 1)
        if not header.startswith("data:image/"):
            raise ValueError("Only image data URLs are accepted.")

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image data.") from exc

    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Face image is too large.")

    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the face image.")
    if image.shape[0] < 160 or image.shape[1] < 160:
        raise ValueError("Face image is too small.")
    return image


def create_embedding(image: np.ndarray) -> np.ndarray:
    detector, recognizer = _load_models()

    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, raw_faces = detector.detect(image)

    faces = filter_faces(raw_faces, width, height, score_threshold=FACE_SCORE_THRESHOLD)

    if len(faces) == 0:
        raise ValueError("No face detected. Center your face and try again.")
    if len(faces) > 1:
        raise ValueError("Multiple faces detected. Only one person may be in the frame.")

    face = faces[0]
    x, y, w, h = [float(v) for v in face[:4]]
    area_ratio = max(w, 0) * max(h, 0) / float(width * height)
    if area_ratio < 0.04:
        raise ValueError("Your face is too small. Move closer to the camera.")

    aligned = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned)
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)

    norm = np.linalg.norm(feature)
    if norm == 0:
        raise ValueError("Could not create a valid face embedding.")
    return feature / norm


def register_face(email: str, image_data: str, overwrite: bool = False) -> dict[str, Any]:
    clean_email = email.strip().lower()
    if not clean_email or "@" not in clean_email:
        raise ValueError("A valid account email is required.")

    if not overwrite and has_face_registered(clean_email):
        raise ValueError("Face ID is already registered for this account.")

    image = decode_image_data(image_data)
    embedding = create_embedding(image)
    ok, message = save_face_embedding(clean_email, embedding)
    if not ok:
        raise RuntimeError(message)

    action = "updated" if overwrite else "registered"
    return {"status": "ok", "message": f"Face ID {action} successfully."}


def match_face(image_data: str) -> dict[str, Any]:
    image = decode_image_data(image_data)
    probe = create_embedding(image)

    candidates = get_face_embeddings_for_matching()
    if not candidates:
        raise LookupError("No Face ID registrations exist yet.")

    best_email = None
    best_score = -1.0

    _, recognizer = _load_models()
    for item in candidates:
        email = str(item.get("email", "")).strip().lower()
        stored = item.get("embedding")
        if not email or not stored:
            continue
        try:
            reference = np.asarray(stored, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(reference)
            if norm == 0 or reference.shape != probe.shape:
                continue
            reference = reference / norm
            score = float(
                recognizer.match(
                    probe.reshape(1, -1),
                    reference.reshape(1, -1),
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
            )
        except Exception:
            # Some OpenCV builds are stricter about the input shape; cosine
            # similarity is equivalent for normalized SFace vectors.
            score = float(np.dot(probe, reference))

        if score > best_score:
            best_score = score
            best_email = email

    if not best_email or best_score < FACE_COSINE_THRESHOLD:
        raise LookupError("Face not recognized.")

    user = get_user_by_email(best_email)
    if not user:
        raise LookupError("The matched face is not linked to a QueryMind account.")

    return {
        "status": "ok",
        "email": best_email,
        "score": round(best_score, 4),
    }
