import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp


@dataclass
class RetouchPreset:
    name: str
    smooth_alpha: float        # сколько сглаживания добавляем (0..1)
    texture_return: float      # сколько возвращаем текстуры (0..1)
    spot_strength: float       # сила точечной чистки (0..1)
    redness_strength: float    # мягкое уменьшение красноты (0..1)
    de_shine_strength: float   # убрать блеск (0..1)


PRESETS = {
    "natural": RetouchPreset(
        "Натурально",
        smooth_alpha=0.28,
        texture_return=0.22,
        spot_strength=0.35,
        redness_strength=0.35,
        de_shine_strength=0.80,
    ),
    "clean": RetouchPreset(
        "Чище кожа",
        smooth_alpha=0.38,
        texture_return=0.22,
        spot_strength=0.55,
        redness_strength=0.50,
        de_shine_strength=0.90,
    ),
    "shine": RetouchPreset(
        "Только убрать блеск",
        smooth_alpha=0.00,
        texture_return=0.00,
        spot_strength=0.00,
        redness_strength=0.15,
        de_shine_strength=0.95,
    ),
}

_mp_face = mp.solutions.face_mesh


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_jpeg_bytes(img_bgr: np.ndarray, quality: int = 96) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _safe_dilate(mask: np.ndarray, px: int) -> np.ndarray:
    k = max(1, int(px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(mask, kernel, iterations=1)


def _feather(mask: np.ndarray, radius: int) -> np.ndarray:
    r = max(1, int(radius))
    return cv2.GaussianBlur(mask, (2 * r + 1, 2 * r + 1), 0)


def _poly_mask(h: int, w: int, points_xy: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points_xy) < 3:
        return mask
    pts = np.array(points_xy, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _clip_point(x, y, w, h):
    return int(max(0, min(w - 1, x))), int(max(0, min(h - 1, y)))


def _face_mesh_masks(img_bgr: np.ndarray):
    """
    Returns:
      face_oval_mask (uint8 0..255),
      eyes_mask (uint8),
      lips_mask (uint8),
      face_bbox (x1,y1,x2,y2) or None
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with _mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        res = face_mesh.process(img_rgb)

    if not res.multi_face_landmarks:
        return None, None, None, None

    lm = res.multi_face_landmarks[0].landmark

    def pt(idx):
        x = int(lm[idx].x * w)
        y = int(lm[idx].y * h)
        return _clip_point(x, y, w, h)

    # Face oval indices (MediaPipe)
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109,
    ]

    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]

    face_oval_mask = _poly_mask(h, w, [pt(i) for i in FACE_OVAL])
    eyes_mask = cv2.bitwise_or(
        _poly_mask(h, w, [pt(i) for i in LEFT_EYE]),
        _poly_mask(h, w, [pt(i) for i in RIGHT_EYE]),
    )
    lips_mask = _poly_mask(h, w, [pt(i) for i in LIPS_OUTER])

    pts = np.array([pt(i) for i in FACE_OVAL], dtype=np.int32)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()

    return face_oval_mask, eyes_mask, lips_mask, (int(x1), int(y1), int(x2), int(y2))


def _skin_mask_in_roi(img_bgr: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Soft skin detection in ROI (neck/decollete fallback). Returns full-size mask."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = roi
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), dtype=np.uint8)

    crop = img_bgr[y1:y2, x1:x2]
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    m = cv2.inRange(ycrcb, lower, upper)

    m = cv2.medianBlur(m, 5)
    m = _feather(m, 6)

    full = np.zeros((h, w), dtype=np.uint8)
    full[y1:y2, x1:x2] = m
    return full


def _de_shine(img_bgr: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """Reduce highlights only (no blur)."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return img_bgr

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    Lf = L.astype(np.float32)

    t = 225.0  # highlight threshold
    highlights = np.clip((Lf - t) / (255.0 - t), 0.0, 1.0)

    m = (mask.astype(np.float32) / 255.0)
    effect = highlights * m * strength

    L_new = Lf * (1.0 - 0.18 * effect)
    L_new = np.clip(L_new, 0, 255).astype(np.uint8)

    out = cv2.merge([L_new, a, b])
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def _soft_smooth(img_bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Skin smoothing with mask + alpha blend."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0:
        return img_bgr

    smooth = cv2.bilateralFilter(img_bgr, d=0, sigmaColor=18, sigmaSpace=8)
    m = (mask.astype(np.float32) / 255.0)[..., None]
    out = img_bgr.astype(np.float32) * (1 - m * alpha) + smooth.astype(np.float32) * (m * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _return_texture(original_bgr: np.ndarray, current_bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Return high-frequency details from original to avoid 'plastic' skin."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return current_bgr

    blur = cv2.GaussianBlur(original_bgr, (0, 0), sigmaX=3.0, sigmaY=3.0)
    texture = original_bgr.astype(np.float32) - blur.astype(np.float32)

    m = (mask.astype(np.float32) / 255.0)[..., None]
    out = current_bgr.astype(np.float32) + texture * (amount * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def _reduce_redness(img_bgr: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return img_bgr

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    af = a.astype(np.float32)
    m = (mask.astype(np.float32) / 255.0)
    neutral = 128.0

    af_new = af - (af - neutral) * (0.18 * strength) * m
    a_new = np.clip(af_new, 0, 255).astype(np.uint8)

    out = cv2.merge([L, a_new, b])
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def _spot_heal(img_bgr: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """Very gentle acne/spot correction."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return img_bgr

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    base = cv2.bilateralFilter(gray, d=0, sigmaColor=20, sigmaSpace=10)
    diff = cv2.absdiff(gray, base)

    thr = int(18 + (1.0 - strength) * 10)  # 18..28
    _, binm = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    binm = cv2.bitwise_and(binm, mask)
    binm = cv2.medianBlur(binm, 5)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    inpaint_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # tiny only -> keep moles/freckles
        if 6 <= area <= 120:
            inpaint_mask[labels == i] = 255

    if inpaint_mask.max() == 0:
        return img_bgr

    radius = 2 if strength < 0.6 else 3
    out = cv2.inpaint(img_bgr, inpaint_mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return out


def retouch_image_bytes(input_bytes: bytes, preset_key: str) -> tuple[bytes, bytes]:
    """
    Returns (after_jpeg_bytes, before_jpeg_bytes)
    """
    preset = PRESETS.get(preset_key, PRESETS["natural"])

    pil = Image.open(io.BytesIO(input_bytes))
    img_bgr = _pil_to_bgr(pil)

    before_jpeg = _bgr_to_jpeg_bytes(img_bgr, quality=96)

    face_oval, eyes, lips, face_bbox = _face_mesh_masks(img_bgr)
    if face_oval is None or face_bbox is None:
        after = _bgr_to_jpeg_bytes(img_bgr, quality=96)
        return after, before_jpeg

    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = face_bbox
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)

    roi_down = int(1.35 * fh)
    roi = (x1 - int(0.15 * fw), y2, x2 + int(0.15 * fw), min(h - 1, y2 + roi_down))
    neck_skin = _skin_mask_in_roi(img_bgr, roi)

    # protect lashes/eyelids with a buffer around eyes
    eyes_buffer = _safe_dilate(eyes, px=int(max(14, fw * 0.06)))
    eyes_buffer = _feather(eyes_buffer, radius=5)

    # brow protection band (heuristic, safe)
    brow_mask = np.zeros((h, w), dtype=np.uint8)
    ex, ey, ew, eh = cv2.boundingRect(eyes)
    if ew > 0 and eh > 0:
        top = max(0, ey - int(0.45 * eh))
        bottom = max(0, ey + int(0.15 * eh))
        left = max(0, ex - int(0.10 * ew))
        right = min(w - 1, ex + ew + int(0.10 * ew))
        brow_mask[top:bottom, left:right] = 255
        brow_mask = cv2.bitwise_and(brow_mask, face_oval)
        brow_mask = _feather(brow_mask, 3)

    base_mask = cv2.bitwise_or(face_oval, neck_skin)
    retouch_mask = base_mask.copy()
    retouch_mask = cv2.subtract(retouch_mask, eyes_buffer.astype(np.uint8))
    retouch_mask = cv2.subtract(retouch_mask, brow_mask.astype(np.uint8))
    retouch_mask = cv2.subtract(retouch_mask, lips.astype(np.uint8))

    retouch_mask = cv2.erode(retouch_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    retouch_mask = _feather(retouch_mask, 4)

    out = img_bgr.copy()
    out = _de_shine(out, retouch_mask, preset.de_shine_strength)
    out = _reduce_redness(out, retouch_mask, preset.redness_strength)
    out = _spot_heal(out, retouch_mask, preset.spot_strength)
    out = _soft_smooth(out, retouch_mask, preset.smooth_alpha)
    out = _return_texture(img_bgr, out, retouch_mask, preset.texture_return)

    after_jpeg = _bgr_to_jpeg_bytes(out, quality=96)
    return after_jpeg, before_jpeg
