import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO

# =======================
# Model paths
# =======================
METAL_MODEL_PATH = "metal_box.pt"
CORRO_MODEL_PATH = "corrosion_seg.pt"

# =======================
# Utilities
# =======================
def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return x1, y1, x2, y2

def union_mask_u8(result, out_h, out_w, mask_thresh=0.45):
    """seg 결과 -> union mask uint8(0/255)"""
    if result.masks is None:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    m = result.masks.data.cpu().numpy()  # (n,h,w)
    union = np.any(m > mask_thresh, axis=0).astype(np.uint8) * 255
    if union.shape[:2] != (out_h, out_w):
        union = cv2.resize(union, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return union

def draw_contours_on_frame(frame_bgr, mask_u8, offset_xy=(0, 0), thickness=2):
    ox, oy = offset_xy
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c + np.array([[[ox, oy]]]) for c in contours]
    cv2.drawContours(frame_bgr, contours, -1, (0, 255, 255), thickness)

def filter_boxes(xyxy, conf, w, h, min_area_ratio=0.02, ar_min=0.25, ar_max=4.0):
    if len(xyxy) == 0:
        return xyxy, conf

    bw = (xyxy[:, 2] - xyxy[:, 0])
    bh = (xyxy[:, 3] - xyxy[:, 1])
    areas = bw * bh
    area_keep = areas > (w * h * min_area_ratio)

    ar = bw / (bh + 1e-6)
    ar_keep = (ar >= ar_min) & (ar <= ar_max)

    keep = area_keep & ar_keep
    return xyxy[keep], conf[keep]

def pick_box_conf_center(xyxy, conf, w, h, center_weight=0.35):
    """score = conf - center_weight * normalized_distance_from_center"""
    cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
    cy = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
    dist = np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
    dist_norm = dist / (np.sqrt((w / 2) ** 2 + (h / 2) ** 2) + 1e-6)

    score = conf - center_weight * dist_norm
    return xyxy[int(np.argmax(score))]

def draw_all_boxes_debug(frame_bgr, xyxy, conf):
    for b, c in zip(xyxy, conf):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame_bgr, f"{c:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

@st.cache_resource
def load_models(metal_path: str, corro_path: str):
    metal = YOLO(metal_path)
    corro = YOLO(corro_path)
    return metal, corro

def read_uploaded_image(uploaded_file) -> np.ndarray:
    """Streamlit 업로드 파일 -> BGR 이미지(OpenCV)"""
    bytes_data = uploaded_file.read()
    nparr = np.frombuffer(bytes_data, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_bgr

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def process_image(
    img_bgr: np.ndarray,
    metal_model,
    corro_model,
    metal_conf: float,
    metal_iou: float,
    metal_imgsz: int,
    use_class_filter: bool,
    metal_class_id: int,
    min_area_ratio: float,
    ar_min: float,
    ar_max: float,
    center_weight: float,
    corro_conf: float,
    corro_imgsz: int,
    mask_thresh: float,
    contour_thickness: int,
    debug_draw_all: bool
):
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # 1) metal detect
    r_m = metal_model.predict(
        out,
        imgsz=metal_imgsz,
        conf=metal_conf,
        iou=metal_iou,
        verbose=False
    )[0]

    if r_m.boxes is None or len(r_m.boxes) == 0:
        return out, None, 0.0, "No METAL box detected"

    xyxy = r_m.boxes.xyxy.cpu().numpy()
    conf = r_m.boxes.conf.cpu().numpy()
    cls = r_m.boxes.cls.cpu().numpy().astype(int)

    if use_class_filter:
        keep = (cls == metal_class_id)
        xyxy = xyxy[keep]
        conf = conf[keep]

    if len(xyxy) == 0:
        return out, None, 0.0, "No METAL boxes after class filter"

    if debug_draw_all:
        draw_all_boxes_debug(out, xyxy, conf)

    # filter size/aspect
    xyxy, conf = filter_boxes(xyxy, conf, w, h, min_area_ratio, ar_min, ar_max)
    if len(xyxy) == 0:
        return out, None, 0.0, "METAL boxes filtered out (size/aspect)"

    # pick main
    main_box = pick_box_conf_center(xyxy, conf, w, h, center_weight)
    box = clamp_box(*main_box, w, h)
    if box is None:
        return out, None, 0.0, "Invalid METAL box"

    x1, y1, x2, y2 = box

    # draw metal box
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(out, "metal(main)", (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # 2) corrosion seg in crop
    crop = out[y1:y2, x1:x2]
    r_c = corro_model.predict(
        crop,
        imgsz=corro_imgsz,
        conf=corro_conf,
        verbose=False
    )[0]
    mask_u8 = union_mask_u8(r_c, crop.shape[0], crop.shape[1], mask_thresh=mask_thresh)

    # 3) raw percent (within bbox)
    raw_pct = float((mask_u8 > 0).mean() * 100.0) if mask_u8.size else 0.0

    # 4) contours
    draw_contours_on_frame(out, mask_u8, offset_xy=(x1, y1), thickness=contour_thickness)

    # 5) text (raw만 표시)
    cv2.putText(out, f"Raw Corrosion%: {raw_pct:.2f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return out, (x1, y1, x2, y2), raw_pct, None

def encode_jpg_bytes(img_bgr: np.ndarray, quality: int = 95) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return buf.tobytes()

# =======================
# Temp/Humi correction
# =======================
def corrected_percent(raw_pct: float, temp_c: float, humi: float,
                      method: str, t0: float, h0: float,
                      a: float, b: float, clamp01: bool = True) -> float:
    """
    method:
      - "mult": final = raw * (1 + a*(H-H0) + b*(T-T0))
      - "add" : final = raw + a*(H-H0) + b*(T-T0)

    a,b는 민감도(튜닝 파라미터).
    """
    dH = humi - h0
    dT = temp_c - t0

    if method == "add":
        out = raw_pct + a * dH + b * dT
    else:
        out = raw_pct * (1.0 + a * dH + b * dT)

    if clamp01:
        out = max(0.0, min(100.0, out))
    return float(out)

# =======================
# Streamlit UI
# =======================
st.set_page_config(page_title="Corrosion% + Temp/Humi Input", layout="wide")
st.title("부식 퍼센티지 측정 + 온습도 입력 보정 (블루투스 없이)")

with st.sidebar:
    st.header("온습도 입력")
    temp_c = st.number_input("온도 (°C)", value=25.0, step=0.5, format="%.1f")
    humi = st.number_input("습도 (%)", value=50.0, step=1.0, format="%.1f")

    st.divider()
    st.header("온습도 보정(캘리브레이션)")
    method = st.selectbox("보정 방식", ["mult", "add"], index=0)
    t0 = st.number_input("기준 온도 T0 (°C)", value=25.0, step=0.5, format="%.1f")
    h0 = st.number_input("기준 습도 H0 (%)", value=50.0, step=1.0, format="%.1f")
    a = st.number_input("습도 민감도 a", value=0.002, step=0.001, format="%.4f",
                        help="mult: (H-H0) 1%당 raw%에 곱해지는 영향")
    b = st.number_input("온도 민감도 b", value=0.002, step=0.001, format="%.4f",
                        help="mult: (T-T0) 1°C당 raw%에 곱해지는 영향")
    clamp_to_0_100 = st.checkbox("최종%를 0~100으로 제한", value=True)

    st.divider()
    st.header("모델/파라미터")

    st.subheader("Metal Detect")
    metal_conf = st.slider("METAL_CONF", 0.05, 0.95, 0.55, 0.01)
    metal_iou = st.slider("METAL_IOU", 0.05, 0.95, 0.50, 0.01)
    metal_imgsz = st.selectbox("METAL_IMGSZ", [320, 416, 512, 640, 800, 960], index=3)

    use_class_filter = st.checkbox("Class filter 사용", value=True)
    metal_class_id = st.number_input("METAL_CLASS_ID", min_value=0, max_value=999, value=0, step=1)

    st.subheader("Metal 박스 필터/선택")
    min_area_ratio = st.slider("MIN_AREA_RATIO (화면 대비 최소 면적)", 0.0, 0.30, 0.02, 0.005)
    ar_min = st.slider("AR_MIN (가로/세로 최소)", 0.05, 3.0, 0.25, 0.05)
    ar_max = st.slider("AR_MAX (가로/세로 최대)", 1.0, 10.0, 4.0, 0.1)
    center_weight = st.slider("CENTER_WEIGHT (중앙 가중치)", 0.0, 1.0, 0.35, 0.05)

    st.subheader("Corrosion Seg")
    corro_conf = st.slider("CORRO_CONF", 0.05, 0.95, 0.25, 0.01)
    corro_imgsz = st.selectbox("CORRO_IMGSZ", [320, 416, 512, 640, 800, 960, 1280], index=5)
    mask_thresh = st.slider("MASK_THRESH", 0.05, 0.95, 0.45, 0.01)

    st.subheader("표시")
    contour_thickness = st.slider("CONTOUR_THICKNESS", 1, 6, 2, 1)
    debug_draw_all = st.checkbox("디버그: 모든 metal 박스(conf) 표시", value=False)

# 모델 로드
metal_model, corro_model = load_models(METAL_MODEL_PATH, CORRO_MODEL_PATH)

uploaded = st.file_uploader("이미지 업로드 (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("왼쪽에서 온습도 입력 후, 이미지를 업로드하면 분석을 시작합니다.")
    st.stop()

img_bgr = read_uploaded_image(uploaded)
if img_bgr is None:
    st.error("이미지를 읽을 수 없습니다. 다른 파일로 시도해보세요.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("원본")
    st.image(bgr_to_rgb(img_bgr), use_container_width=True)

# 처리 (raw% 계산)
result_bgr, metal_box, raw_pct, err = process_image(
    img_bgr=img_bgr,
    metal_model=metal_model,
    corro_model=corro_model,
    metal_conf=metal_conf,
    metal_iou=metal_iou,
    metal_imgsz=metal_imgsz,
    use_class_filter=use_class_filter,
    metal_class_id=int(metal_class_id),
    min_area_ratio=min_area_ratio,
    ar_min=ar_min,
    ar_max=ar_max,
    center_weight=center_weight,
    corro_conf=corro_conf,
    corro_imgsz=corro_imgsz,
    mask_thresh=mask_thresh,
    contour_thickness=contour_thickness,
    debug_draw_all=debug_draw_all,
)

# 최종% (온습도 보정)
final_pct = corrected_percent(
    raw_pct=raw_pct,
    temp_c=float(temp_c),
    humi=float(humi),
    method=method,
    t0=float(t0),
    h0=float(h0),
    a=float(a),
    b=float(b),
    clamp01=clamp_to_0_100
)

with col2:
    st.subheader("결과")
    st.image(bgr_to_rgb(result_bgr), use_container_width=True)

    if err:
        st.error(err)
    else:
        st.success(f"Raw Corrosion% (metal bbox 기준): {raw_pct:.2f}%")
        st.info(f"입력 온습도: T={float(temp_c):.1f}°C, H={float(humi):.1f}%")
        st.success(f"Final Corrosion% (온습도 보정): {final_pct:.2f}%")
        st.write(f"Metal box: {metal_box}")
        st.caption(f"보정식: {method}, 기준(T0={t0}, H0={h0}), a={a}, b={b}")

    # 다운로드
    jpg_bytes = encode_jpg_bytes(result_bgr, quality=95)
    st.download_button(
        label="결과 이미지 다운로드 (JPG)",
        data=jpg_bytes,
        file_name="corrosion_result.jpg",
        mime="image/jpeg",
        use_container_width=True
    )
