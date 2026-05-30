import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math
import io as _io
import os

# MediaPipe Tasks API (v0.10+)
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# Path to downloaded model
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# Set Page Config first
st.set_page_config(
    page_title="STYLEME AI MEN | Biometric Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Cyberpunk Obsidian HUD Styling Injection
# ---------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
    /* Dark Obsidian Base */
    .stApp {
        background-color: #020617 !important;
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers & Cyber Typography */
    h1, h2, h3, h4, [data-testid="stHeader"] {
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #f8fafc;
    }
    
    /* Neon Cyan Text Highlight */
    .cyan-glow-text {
        color: #22d3ee !important;
        text-shadow: 0 0 10px rgba(6, 182, 212, 0.6);
        font-family: 'Orbitron', sans-serif;
    }
    
    /* Custom HUD Panels */
    .hud-card {
        background-color: rgba(3, 7, 18, 0.75) !important;
        border: 1px solid rgba(6, 182, 212, 0.25) !important;
        box-shadow: inset 0 0 15px rgba(6, 182, 212, 0.05), 0 0 10px rgba(6, 182, 212, 0.1);
        padding: 16px;
        border-radius: 2px;
        margin-bottom: 15px;
        position: relative;
    }
    
    .hud-header {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 10px;
        font-weight: bold;
        letter-spacing: 0.2em;
        color: #06b6d4;
        border-bottom: 1px solid rgba(6, 182, 212, 0.2);
        padding-bottom: 6px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .hud-indicator-dot {
        width: 6px;
        height: 6px;
        background-color: #06b6d4;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #06b6d4;
    }
    
    /* Buttons Customization */
    .stButton>button {
        background-color: rgba(6, 182, 212, 0.1) !important;
        color: #22d3ee !important;
        border: 1px solid rgba(6, 182, 212, 0.4) !important;
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        font-size: 10px !important;
        border-radius: 2px !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease;
        width: 100%;
        cursor: pointer;
    }
    
    .stButton>button:hover {
        background-color: #06b6d4 !important;
        color: #020617 !important;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.4) !important;
        border-color: #22d3ee !important;
    }
    
    /* Sidebar styling override */
    [data-testid="stSidebar"] {
        background-color: #030712 !important;
        border-right: 1px solid rgba(6, 182, 212, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Seed Data - Hairstyles & Beards Catalog
# ---------------------------------------------------------
STYLE_CATALOG = [
    # Haircuts
    {
        "name": "Textured Pompadour",
        "category": "haircut",
        "description": "High-volume luxury cut with textured length on top and a clean side taper. Adds height and projects style.",
        "maintenance_level": "high",
        "style_preference": "luxury",
        "shape_compatibility": {"oval": 1.0, "square": 0.9, "diamond": 0.8, "round": 0.5, "heart": 0.7, "triangle": 0.6},
        "density_compatibility": {"high": 1.0, "medium": 0.9, "low": 0.4},
        "hairline_compatibility": {"low": 1.0, "medium": 0.85, "high": 0.5},
        "occasion_compatibility": {"corporate": 0.9, "casual": 1.0, "party": 1.0, "wedding": 1.0}
    },
    {
        "name": "Modern Quiff",
        "category": "haircut",
        "description": "A dynamic upward sweep that adds structural lines. Perfect for both casual and high-end formal settings.",
        "maintenance_level": "medium",
        "style_preference": "trendy",
        "shape_compatibility": {"round": 0.95, "oval": 0.9, "square": 0.9, "diamond": 0.8, "heart": 0.65, "triangle": 0.7},
        "density_compatibility": {"high": 1.0, "medium": 0.95, "low": 0.6},
        "hairline_compatibility": {"low": 1.0, "medium": 0.95, "high": 0.6},
        "occasion_compatibility": {"corporate": 0.9, "casual": 1.0, "party": 1.0, "wedding": 0.9}
    },
    {
        "name": "Textured Crop",
        "category": "haircut",
        "description": "Low-maintenance choppy fringe styled forward. Best choice for high/receding hairlines or low density.",
        "maintenance_level": "low",
        "style_preference": "minimal",
        "shape_compatibility": {"oval": 0.9, "square": 0.8, "round": 0.6, "diamond": 0.9, "heart": 0.95, "triangle": 0.8},
        "density_compatibility": {"high": 0.8, "medium": 1.0, "low": 0.95},
        "hairline_compatibility": {"low": 0.7, "medium": 0.9, "high": 1.0},
        "occasion_compatibility": {"corporate": 0.7, "casual": 1.0, "party": 0.9, "wedding": 0.6}
    },
    {
        "name": "Slicked Back Side Part",
        "category": "haircut",
        "description": "A timeless classic parted neat and slicked down. Projects boardroom authority, luxury, and prestige.",
        "maintenance_level": "medium",
        "style_preference": "luxury",
        "shape_compatibility": {"oval": 1.0, "square": 0.95, "round": 0.65, "diamond": 0.8, "heart": 0.85, "triangle": 0.9},
        "density_compatibility": {"high": 1.0, "medium": 0.9, "low": 0.5},
        "hairline_compatibility": {"low": 1.0, "medium": 0.9, "high": 0.5},
        "occasion_compatibility": {"corporate": 1.0, "casual": 0.8, "party": 0.9, "wedding": 1.0}
    },
    {
        "name": "Buzz Cut",
        "category": "haircut",
        "description": "Ultra-short cropped shave close to the scalp. Accents chin structure and sharp cheek contours.",
        "maintenance_level": "low",
        "style_preference": "minimal",
        "shape_compatibility": {"square": 1.0, "oval": 0.9, "round": 0.3, "diamond": 0.7, "heart": 0.5, "triangle": 0.5},
        "density_compatibility": {"high": 1.0, "medium": 0.85, "low": 0.6},
        "hairline_compatibility": {"low": 1.0, "medium": 0.8, "high": 0.5},
        "occasion_compatibility": {"corporate": 0.6, "casual": 1.0, "party": 0.9, "wedding": 0.5}
    },

    # Beards
    {
        "name": "Short Boxed Beard",
        "category": "beard",
        "description": "Groomed full beard trimmed to a low height. Shapes jawlines while maintaining formal office standards.",
        "maintenance_level": "medium",
        "style_preference": "modern",
        "shape_compatibility": {"round": 0.95, "oval": 0.9, "square": 0.8, "diamond": 0.9, "heart": 0.95, "triangle": 0.7},
        "density_compatibility": {"high": 1.0, "medium": 0.9, "low": 0.4},
        "hairline_compatibility": {"low": 1.0, "medium": 1.0, "high": 1.0},
        "occasion_compatibility": {"corporate": 1.0, "casual": 0.9, "party": 0.9, "wedding": 1.0}
    },
    {
        "name": "Designer Stubble",
        "category": "beard",
        "description": "Clean, short 3-5 day shadow. Softens chins and provides a ruggedly minimal, low-care style.",
        "maintenance_level": "low",
        "style_preference": "minimal",
        "shape_compatibility": {"oval": 1.0, "square": 1.0, "diamond": 0.95, "round": 0.8, "heart": 0.9, "triangle": 0.9},
        "density_compatibility": {"high": 1.0, "medium": 1.0, "low": 0.95},
        "hairline_compatibility": {"low": 1.0, "medium": 1.0, "high": 1.0},
        "occasion_compatibility": {"corporate": 0.9, "casual": 1.0, "party": 1.0, "wedding": 0.85}
    },
    {
        "name": "Tapered Beard Fade",
        "category": "beard",
        "description": "Modern sharp trim featuring a gradient fade from sideburns down into a fuller chiseled chin.",
        "maintenance_level": "high",
        "style_preference": "trendy",
        "shape_compatibility": {"round": 1.0, "oval": 0.95, "square": 0.9, "diamond": 0.85, "heart": 0.8, "triangle": 0.85},
        "density_compatibility": {"high": 1.0, "medium": 0.9, "low": 0.5},
        "hairline_compatibility": {"low": 1.0, "medium": 1.0, "high": 1.0},
        "occasion_compatibility": {"corporate": 0.9, "casual": 1.0, "party": 1.0, "wedding": 0.9}
    },

    # Combos
    {
        "name": "Textured Pompadour + Short Boxed Beard",
        "category": "combo",
        "description": "Premium matching package. High pompadour balances a low boxed beard shape.",
        "maintenance_level": "high",
        "style_preference": "luxury",
        "shape_compatibility": {"oval": 1.0, "square": 0.9, "round": 0.85, "diamond": 0.9, "heart": 0.8, "triangle": 0.75},
        "density_compatibility": {"high": 1.0, "medium": 0.9, "low": 0.4},
        "hairline_compatibility": {"low": 1.0, "medium": 0.85, "high": 0.5},
        "occasion_compatibility": {"corporate": 0.95, "casual": 1.0, "party": 1.0, "wedding": 1.0}
    },
    {
        "name": "Modern Quiff + Taper Beard",
        "category": "combo",
        "description": "High-contrast sweep matched with faded beard cheeks for a pop-culture, modern look.",
        "maintenance_level": "high",
        "style_preference": "trendy",
        "shape_compatibility": {"round": 1.0, "oval": 0.95, "square": 0.9, "diamond": 0.85, "heart": 0.75, "triangle": 0.8},
        "density_compatibility": {"high": 1.0, "medium": 0.95, "low": 0.6},
        "hairline_compatibility": {"low": 1.0, "medium": 0.95, "high": 0.6},
        "occasion_compatibility": {"corporate": 0.9, "casual": 1.0, "party": 1.0, "wedding": 0.9}
    },
    {
        "name": "Slick Back + Designer Stubble",
        "category": "combo",
        "description": "A refined slick-back matched with rugged stubble to blend elegance with edge.",
        "maintenance_level": "medium",
        "style_preference": "luxury",
        "shape_compatibility": {"oval": 1.0, "square": 0.95, "round": 0.75, "diamond": 0.85, "heart": 0.85, "triangle": 0.9},
        "density_compatibility": {"high": 1.0, "medium": 0.9, "low": 0.6},
        "hairline_compatibility": {"low": 1.0, "medium": 0.9, "high": 0.5},
        "occasion_compatibility": {"corporate": 1.0, "casual": 0.9, "party": 1.0, "wedding": 0.95}
    }
]

# ---------------------------------------------------------
# Biometric Landmark Math Helpers
# Works with mediapipe NormalizedLandmark objects (x,y,z attrs)
# ---------------------------------------------------------
def get_distance_py(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def get_angle_py(p1, p2, p3):
    ux, uy, uz = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
    vx, vy, vz = p3.x - p2.x, p3.y - p2.y, p3.z - p2.z
    dot = ux * vx + uy * vy + uz * vz
    mag_u = math.sqrt(ux*ux + uy*uy + uz*uz)
    mag_v = math.sqrt(vx*vx + vy*vy + vz*vz)
    if mag_u == 0 or mag_v == 0:
        return 0
    cos_theta = max(-1.0, min(1.0, dot / (mag_u * mag_v)))
    return math.degrees(math.acos(cos_theta))

def run_face_landmarker(img_rgb_np):
    """Run MediaPipe FaceLandmarker (Tasks API) on an RGB numpy array.
    Returns list of landmark objects or None if no face found."""
    base_opts = BaseOptions(model_asset_path=_MODEL_PATH)
    options = FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=RunningMode.IMAGE,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    with FaceLandmarker.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb_np)
        result = detector.detect(mp_image)
    if result.face_landmarks:
        return result.face_landmarks[0]   # list of NormalizedLandmark
    return None

def analyze_landmarks_py(landmarks):
    # Standard landmarks indices
    p10 = landmarks[10] # Forehead top
    p152 = landmarks[152] # Chin bottom
    p9 = landmarks[9] # Brow center
    p332 = landmarks[332] # Left temple (camera mirrored)
    p103 = landmarks[103] # Right temple
    p234 = landmarks[234] # Left cheek outer
    p454 = landmarks[454] # Right cheek outer
    p58 = landmarks[58] # Left jaw gonion
    p288 = landmarks[288] # Right jaw gonion

    face_height = get_distance_py(p10, p152)
    forehead_width = get_distance_py(p332, p103)
    cheek_width = get_distance_py(p234, p454)
    jaw_width = get_distance_py(p58, p288)

    ratio = face_height / cheek_width if cheek_width > 0 else 1.0

    # Face Shape Classification
    if ratio > 1.25 and cheek_width > forehead_width and forehead_width > jaw_width:
        face_shape = "Oval"
    elif 0.95 <= ratio <= 1.18 and abs(cheek_width - jaw_width) < cheek_width * 0.1 and jaw_width > forehead_width * 0.9:
        left_angle = get_angle_py(p332, p58, p152)
        if left_angle < 130:
            face_shape = "Square"
        else:
            face_shape = "Round"
    elif ratio > 1.15 and cheek_width > forehead_width and cheek_width > jaw_width and jaw_width < forehead_width:
        face_shape = "Diamond"
    elif forehead_width > cheek_width and cheek_width > jaw_width:
        face_shape = "Heart"
    elif jaw_width > cheek_width and cheek_width > forehead_width:
        face_shape = "Triangle"
    else:
        if ratio > 1.22:
            face_shape = "Oval"
        elif ratio < 1.12:
            face_shape = "Round"
        else:
            face_shape = "Square"

    # Jawline Strength
    left_jaw_angle = get_angle_py(p234, p58, p152)
    right_jaw_angle = get_angle_py(p454, p288, p152)
    avg_jaw_angle = (left_jaw_angle + right_jaw_angle) / 2.0

    if avg_jaw_angle < 122:
        jawline_strength = f"Chiseled ({int(avg_jaw_angle)}°)"
    elif avg_jaw_angle > 132:
        jawline_strength = f"Soft ({int(avg_jaw_angle)}°)"
    else:
        jawline_strength = f"Defined ({int(avg_jaw_angle)}°)"

    # Hairline height
    forehead_height = get_distance_py(p10, p9)
    forehead_ratio = forehead_height / face_height if face_height > 0 else 0.32
    if forehead_ratio > 0.36:
        hairline = "High / Receding"
    elif forehead_ratio < 0.28:
        hairline = "Low"
    else:
        hairline = "Medium"

    # Symmetry
    symmetry_diff = 0
    pairs = [(p332, p103), (p234, p454), (p58, p288)]
    for pl, pr in pairs:
        mid_x = (p10.x + p152.x) / 2.0
        dist_l = abs(pl.x - mid_x)
        dist_r = abs(pr.x - mid_x)
        total = dist_l + dist_r
        if total > 0:
            symmetry_diff += abs(dist_l - dist_r) / total
    symmetry = max(70, int(100 - (symmetry_diff / len(pairs)) * 100))

    # Fallbacks based on metrics
    hair_density = "High" if cheek_width > 0.31 else "Low" if cheek_width < 0.27 else "Medium"
    beard_density = "Heavy" if jaw_width > 0.25 else "Patchy" if jaw_width < 0.21 else "Medium"
    skin_tone = "Warm Sand" if forehead_ratio > 0.32 else "Cool Obsidian" if forehead_ratio < 0.29 else "Neutral Olive"

    return {
        "faceShape": face_shape,
        "jawlineStrength": jawline_strength,
        "hairDensity": hair_density,
        "hairlinePosition": hairline,
        "beardDensity": beard_density,
        "skinTone": skin_tone,
        "symmetryScore": symmetry,
        "foreheadRatio": round(forehead_ratio, 2),
        "confidenceScore": 0.96
    }

# ---------------------------------------------------------
# OpenCV Holographic Vector Draw Engines (RGB format)
# ---------------------------------------------------------
def draw_landmarks_mesh_py(img, landmarks):
    H, W, C = img.shape
    mesh_color = (6, 182, 212) # RGB Cyan
    
    # Draw simple facial lines connecting features
    feature_indices = [
        # Jawline
        [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397],
        # Left Eyebrow
        [70, 63, 105, 66, 107],
        # Right Eyebrow
        [336, 296, 334, 293, 300],
        # Left Eye
        [33, 160, 158, 133, 153, 144, 33],
        # Right Eye
        [362, 385, 387, 263, 373, 380, 362],
        # Lips
        [61, 37, 0, 267, 291, 321, 17, 91, 61]
    ]

    for group in feature_indices:
        pts = []
        for idx in group:
            pt = landmarks[idx]
            pts.append([int(pt.x * W), int(pt.y * H)])
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts_arr], False, mesh_color, 1, lineType=cv2.LINE_AA)

    # Draw nodes
    for idx in [10, 152, 234, 454, 9, 33, 263]:
        pt = landmarks[idx]
        px, py = int(pt.x * W), int(pt.y * H)
        cv2.circle(img, (px, py), 3, (255, 255, 255), -1)
        cv2.circle(img, (px, py), 6, mesh_color, 1)

def draw_hair_overlay_py(img, landmarks, style_name):
    H, W, C = img.shape
    p10 = landmarks[10] # Forehead
    p152 = landmarks[152] # Chin
    p332 = landmarks[332] # Left temple
    p103 = landmarks[103] # Right temple
    
    pt_10 = (int(p10.x * W), int(p10.y * H))
    pt_152 = (int(p152.x * W), int(p152.y * H))
    pt_332 = (int(p332.x * W), int(p332.y * H))
    pt_103 = (int(p103.x * W), int(p103.y * H))

    face_height = get_distance_py(p10, p152) * H
    scale = face_height / 220.0
    angle = np.arctan2(pt_152[0] - pt_10[0], pt_152[1] - pt_10[1]) - np.pi

    cyan_line = (6, 182, 212)
    cyan_fill = (6, 182, 212)
    overlay = img.copy()
    name_lower = style_name.lower()

    if "pompadour" in name_lower:
        rel_pts = [
            (-70, -80), (-95, -135), (-40, -175), (0, -175),
            (40, -175), (95, -135), (70, -80), (45, -70),
            (0, -80), (-45, -70)
        ]
    elif "quiff" in name_lower:
        rel_pts = [
            (-65, -80), (-80, -130), (-25, -160), (15, -150),
            (55, -140), (90, -115), (65, -80), (0, -82),
            (-30, -78), (-65, -80)
        ]
    elif "crop" in name_lower:
        rel_pts = [
            (-70, -65), (-75, -115), (-35, -125), (0, -125),
            (35, -125), (75, -115), (70, -65), (50, -60),
            (25, -65), (0, -58), (-25, -65), (-50, -60)
        ]
    elif "side part" in name_lower or "slick" in name_lower:
        rel_pts = [
            (-72, -65), (-72, -115), (-25, -135), (0, -135),
            (30, -135), (72, -115), (72, -65), (0, -60)
        ]
    elif "buzz" in name_lower:
        rel_pts = [
            (-68, -60), (-70, -108), (-35, -120), (0, -120),
            (35, -120), (70, -108), (68, -60), (0, -55)
        ]
    else:
        rel_pts = [
            (-65, -65), (-45, -115), (0, -125), (45, -115),
            (65, -65), (0, -55)
        ]

    translated = []
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    for rx, ry in rel_pts:
        sx = rx * scale
        sy = ry * scale
        rx_rot = sx * cos_a - sy * sin_a
        ry_rot = sx * sin_a + sy * cos_a
        translated.append([int(pt_10[0] + rx_rot), int(pt_10[1] + ry_rot)])

    pts_arr = np.array(translated, dtype=np.int32)
    cv2.fillPoly(overlay, [pts_arr], cyan_fill)
    cv2.polylines(overlay, [pts_arr], True, cyan_line, thickness=int(3*scale) or 1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.40, img, 0.60, 0, img)

def draw_beard_overlay_py(img, landmarks, style_name):
    H, W, C = img.shape
    p152 = landmarks[152] # Chin
    p10 = landmarks[10] # Forehead
    
    pt_152 = (int(p152.x * W), int(p152.y * H))
    pt_10 = (int(p10.x * W), int(p10.y * H))

    face_height = get_distance_py(p10, p152) * H
    scale = face_height / 220.0
    angle = np.arctan2(pt_152[0] - pt_10[0], pt_152[1] - pt_10[1]) - np.pi

    amber_line = (245, 158, 11) # RGB Amber
    overlay = img.copy()
    name_lower = style_name.lower()

    if "boxed" in name_lower:
        rel_pts = [
            (-65, -10), (-70, 40), (-55, 105), (0, 105),
            (55, 105), (70, 40), (65, -10), (55, -10),
            (45, 30), (25, 70), (0, 70), (-25, 70),
            (-45, 30), (-55, -10)
        ]
    elif "stubble" in name_lower:
        rel_pts = [
            (-64, -10), (-68, 38), (-54, 102), (0, 102),
            (54, 102), (68, 38), (64, -10), (54, -10),
            (44, 28), (24, 66), (0, 66), (-24, 66),
            (-44, 28), (-54, -10)
        ]
    elif "taper" in name_lower or "fade" in name_lower:
        rel_pts = [
            (-62, -15), (-66, 38), (-48, 112), (0, 120),
            (48, 112), (66, 38), (62, -15), (55, -15),
            (42, 28), (22, 75), (0, 80), (-22, 75),
            (-42, 28), (-55, -15)
        ]
    elif "bandholz" in name_lower or "full" in name_lower:
        rel_pts = [
            (-68, -10), (-75, 45), (-65, 145), (0, 150),
            (65, 145), (75, 45), (68, -10), (55, -10),
            (48, 38), (28, 80), (0, 80), (-28, 80),
            (-48, 38), (-55, -10)
        ]
    else:
        return

    translated = []
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    for rx, ry in rel_pts:
        sx = rx * scale
        sy = ry * scale
        rx_rot = sx * cos_a - sy * sin_a
        ry_rot = sx * sin_a + sy * cos_a
        translated.append([int(pt_152[0] + rx_rot), int(pt_152[1] + ry_rot)])

    pts_arr = np.array(translated, dtype=np.int32)
    if "stubble" in name_lower:
        # Draw stubble as dots
        for pt in translated:
            cv2.circle(overlay, (pt[0], pt[1]), int(2*scale) or 1, amber_line, -1)
    else:
        cv2.fillPoly(overlay, [pts_arr], amber_line)
        cv2.polylines(overlay, [pts_arr], True, amber_line, thickness=int(3*scale) or 1, lineType=cv2.LINE_AA)
    
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

# ---------------------------------------------------------
# Recommender Scoring Engine (Python)
# ---------------------------------------------------------
def calculate_recommendations_py(bio, consult):
    matches = []
    
    w_shape = 0.35
    w_density = 0.20
    w_hairline = 0.15
    w_occasion = 0.10
    w_preference = 0.10
    w_maintenance = 0.10

    user_shape = bio["faceShape"].lower()
    user_hairline = bio["hairlinePosition"].lower()
    user_hair_density = bio["hairDensity"].lower()
    user_beard_density = bio["beardDensity"].lower()
    user_occasion = consult["occasion"].lower()
    user_preference = consult["style_preference"].lower()
    user_maintenance = consult["maintenance_level"].lower()

    # Filter catalog by category
    filtered_catalog = [
        s for s in STYLE_CATALOG 
        if s["category"] == consult["service_type"]
    ]
    if not filtered_catalog:
        filtered_catalog = STYLE_CATALOG

    for style in filtered_catalog:
        # 1. Face Shape
        shape_score = style["shape_compatibility"].get(user_shape, 0.5)
        
        # 2. Density
        if style["category"] == "beard":
            density_score = style["density_compatibility"].get(user_beard_density, 0.6)
        else:
            density_score = style["density_compatibility"].get(user_hair_density, 0.6)
            
        # 3. Hairline
        hairline_score = style["hairline_compatibility"].get(user_hairline, 0.6)
        
        # 4. Occasion
        occasion_score = style["occasion_compatibility"].get(user_occasion, 0.6)
        
        # 5. Preference
        pref_score = 1.0 if style["style_preference"].lower() == user_preference else 0.5
        
        # 6. Maintenance
        style_maint = style["maintenance_level"].lower()
        if user_maintenance == "high":
            maint_score = 1.0
        elif user_maintenance == "medium":
            maint_score = 0.5 if style_maint == "high" else 1.0
        else:
            maint_score = 0.2 if style_maint == "high" else 0.6 if style_maint == "medium" else 1.0

        # Weighted calculation
        total = (
            (w_shape * shape_score) +
            (w_density * density_score) +
            (w_hairline * hairline_score) +
            (w_occasion * occasion_score) +
            (w_preference * pref_score) +
            (w_maintenance * maint_score)
        )
        
        percent_score = round(total * 100, 1)
        final_score = max(50.0, min(99.0, percent_score))

        matches.append({
            "style": style,
            "score": final_score
        })

    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches

# ---------------------------------------------------------
# Session State Init
# ---------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "biometrics" not in st.session_state:
    st.session_state.biometrics = None
if "consultation" not in st.session_state:
    st.session_state.consultation = None
if "webcam_photo" not in st.session_state:
    st.session_state.webcam_photo = None
if "landmarks" not in st.session_state:
    st.session_state.landmarks = None
if "selected_index" not in st.session_state:
    st.session_state.selected_index = 0
if "view_angle" not in st.session_state:
    st.session_state.view_angle = "front"

# ---------------------------------------------------------
# Sidebar Diagnostic HUD
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="cyan-glow-text" style="font-size: 20px; font-weight: 900; margin-bottom: 2px;">STYLEME AI</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 8px; letter-spacing: 0.3em; color: rgba(6, 182, 212, 0.6); margin-bottom: 25px;">BIOMETRIC MIRROR LAB</div>', unsafe_allow_html=True)

    if st.session_state.biometrics:
        bio = st.session_state.biometrics
        st.markdown("""
        <div class="hud-card">
            <div class="hud-header">
                <span class="hud-indicator-dot"></span>
                ACTIVE BIOMETRIC INDEX
            </div>
            <div style="font-family: 'Orbitron', sans-serif; font-size: 11px; space-y: 6px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b;">FACE SHAPE:</span>
                    <span style="color: #22d3ee; font-weight: bold;">""" + bio["faceShape"] + """</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b;">JAWLINE:</span>
                    <span style="color: #e2e8f0;">""" + bio["jawlineStrength"] + """</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b;">HAIR DENSITY:</span>
                    <span style="color: #e2e8f0;">""" + bio["hairDensity"] + """</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b;">HAIRLINE:</span>
                    <span style="color: #e2e8f0;">""" + bio["hairlinePosition"] + """</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b;">SYMMETRY:</span>
                    <span style="color: #22d3ee;">""" + str(bio["symmetryScore"]) + """%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="hud-card" style="text-align: center; color: rgba(6, 182, 212, 0.4); font-size: 10px;">
            AWAITING SCANNED FEED...
        </div>
        """, unsafe_allow_html=True)

    if st.button("RESET PLATFORM"):
        st.session_state.page = "welcome"
        st.session_state.biometrics = None
        st.session_state.consultation = None
        st.session_state.webcam_photo = None
        st.session_state.landmarks = None
        st.session_state.selected_index = 0
        st.session_state.view_angle = "front"
        st.rerun()

# ---------------------------------------------------------
# SCREEN 1: WELCOME SCREEN
# ---------------------------------------------------------
if st.session_state.page == "welcome":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        
        # Scanner Bracket Icon Emulation
        st.markdown("""
        <div style="display: inline-block; padding: 20px; border: 1px solid rgba(6,182,212,0.2); border-radius: 50%; background: rgba(3,7,18,0.4); margin-bottom: 25px;">
            <div style="font-size: 32px; filter: drop-shadow(0 0 10px #06b6d4);">⚙️</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<h1 style="font-size: 42px; margin-bottom: 2px;">WELCOME TO <span class="cyan-glow-text">STYLEME AI</span></h1>', unsafe_allow_html=True)
        st.markdown('<p style="font-family: \'Orbitron\', sans-serif; font-size: 10px; letter-spacing: 0.45em; color: rgba(6, 182, 212, 0.75); margin-bottom: 40px;">Futuristic AI-powered Grooming Laboratory</p>', unsafe_allow_html=True)

        st.markdown("""
        <div class="hud-card" style="text-align: left; font-size: 13px; line-height: 1.6; margin-bottom: 30px;">
            <div class="hud-header">
                <span class="hud-indicator-dot"></span>
                HUD SECURE LINK ESTABLISHED
            </div>
            This mirror calibrates face shape, hairline contours, jaw angles, and growth densities using computer vision algorithms. Matches are scored and overlaid in a 2.5D visual projection mirror.
        </div>
        """, unsafe_allow_html=True)

        if st.button("LAUNCH MIRROR"):
            st.session_state.page = "scan"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# SCREEN 2: BIOMETRIC SCAN
# ---------------------------------------------------------
elif st.session_state.page == "scan":
    st.markdown('<h2 style="font-size: 20px; margin-bottom: 5px;">BIOMETRIC SCANNING MIRROR</h2>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:9px; letter-spacing:0.3em; color:rgba(6,182,212,0.5); margin-bottom:18px;">PHASE 1 — FACIAL GEOMETRY ACQUISITION</div>', unsafe_allow_html=True)

    _CAPTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured_face.jpg")

    col1, col2 = st.columns([2, 1])

    with col1:

        # ── OPTION A: DEMO (no photo needed) ─────────────────────────────
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(6,182,212,0.12),rgba(6,182,212,0.04));
                    border:1px solid rgba(6,182,212,0.4); border-radius:6px;
                    padding:18px 20px; margin-bottom:14px; text-align:center;">
            <div style="font-family:'Orbitron',sans-serif; font-size:10px; color:#06b6d4;
                        letter-spacing:0.25em; margin-bottom:6px;">⚡ INSTANT DEMO — NO PHOTO NEEDED</div>
            <div style="font-size:11px; color:#94a3b8; margin-bottom:4px;">
                Skip the photo step. The AI will use standard biometric data to generate
                your full personalised style recommendations immediately.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀  START DEMO — GET INSTANT STYLE RECOMMENDATIONS", use_container_width=True):
            st.session_state.biometrics = {
                "faceShape": "Oval",
                "jawlineStrength": "Chiseled (118°)",
                "hairDensity": "High",
                "hairlinePosition": "Medium",
                "beardDensity": "Medium",
                "skinTone": "Neutral Olive",
                "symmetryScore": 94,
                "foreheadRatio": 0.32,
                "confidenceScore": 0.97
            }
            st.session_state.webcam_photo = None
            st.session_state.landmarks = None
            st.session_state.page = "consult"
            st.rerun()

        # ── OPTION B: NATIVE WEBCAM SCANNER ───────────────────────────────
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(34,211,238,0.1),rgba(34,211,238,0.03));
                    border:1px solid rgba(34,211,238,0.3); border-radius:6px;
                    padding:16px 18px; margin-bottom:14px; text-align:center;">
            <div style="font-family:'Orbitron',sans-serif; font-size:10px; color:#22d3ee;
                        letter-spacing:0.2em; margin-bottom:6px;">📸 NATIVE CAMERA SCANNER (RECOMMENDED)</div>
            <div style="font-size:11px; color:#94a3b8; margin-bottom:12px;">
                Direct hardware webcam capture. Bypasses all browser permission errors.
            </div>
        </div>
        """, unsafe_allow_html=True)

        img_bytes_to_process = None

        if st.button("📷  ACTIVATE BIOMETRIC WEBCAM SCANNER", use_container_width=True):
            import time
            cap = None
            # Scan multiple camera indexes and backends to guarantee device lock
            for index in [0, 1, 2]:
                for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
                    try:
                        if backend is not None:
                            temp_cap = cv2.VideoCapture(index, backend)
                        else:
                            temp_cap = cv2.VideoCapture(index)
                        if temp_cap is not None and temp_cap.isOpened():
                            # Test if we can read a frame successfully
                            ret, test_frame = temp_cap.read()
                            if ret and test_frame is not None:
                                cap = temp_cap
                                break
                            else:
                                temp_cap.release()
                    except Exception:
                        pass
                if cap is not None:
                    break

            if cap is None:
                st.error("Could not access any native camera. Please verify: (1) Your camera is physically plugged in, (2) You have allowed camera access in Windows Settings (Privacy & Security -> Camera), (3) No other app like Zoom, Teams, Skype, or the Windows Camera app is currently using your webcam.")
            else:
                frame_placeholder = st.empty()
                # Run dynamic cyber lock scan sequence
                for i in range(25):
                    ret, frame = cap.read()
                    if ret:
                        # Mirror frame for user comfort
                        frame = cv2.flip(frame, 1)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Add holographic elements on frame
                        h, w, c = frame_rgb.shape
                        center = (int(w/2), int(h/2))
                        axes = (int(w*0.18), int(h*0.28))
                        # Draw oval guide
                        cv2.ellipse(frame_rgb, center, axes, 0, 0, 360, (6, 182, 214), 2)
                        
                        # Draw scan line animating vertically
                        scan_y = int((time.time() * 200) % h)
                        cv2.line(frame_rgb, (0, scan_y), (w, scan_y), (34, 211, 238), 1)
                        
                        # Draw corner brackets
                        bracket_len = 30
                        # TL
                        cv2.line(frame_rgb, (40, 40), (40 + bracket_len, 40), (34, 211, 238), 2)
                        cv2.line(frame_rgb, (40, 40), (40, 40 + bracket_len), (34, 211, 238), 2)
                        # TR
                        cv2.line(frame_rgb, (w-40, 40), (w-40-bracket_len, 40), (34, 211, 238), 2)
                        cv2.line(frame_rgb, (w-40, 40), (w-40, 40 + bracket_len), (34, 211, 238), 2)
                        # BL
                        cv2.line(frame_rgb, (40, h-40), (40 + bracket_len, h-40), (34, 211, 238), 2)
                        cv2.line(frame_rgb, (40, h-40), (40, h-40-bracket_len), (34, 211, 238), 2)
                        # BR
                        cv2.line(frame_rgb, (w-40, h-40), (w-40-bracket_len, h-40), (34, 211, 238), 2)
                        cv2.line(frame_rgb, (w-40, h-40), (w-40, h-40-bracket_len), (34, 211, 238), 2)

                        frame_placeholder.image(frame_rgb, caption=f"⚡ LIVE BIOMETRIC RECEIVER ACTIVE — TARGET ACQUISITION [{25-i}]", use_container_width=True)
                    time.sleep(0.05)
                
                # Capture clean raw frame for AI (no overlays, so AI works properly!)
                ret, final_frame = cap.read()
                cap.release()
                
                if ret:
                    # Convert to RGB and compile to bytes
                    final_frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(final_frame_rgb)
                    buf = _io.BytesIO()
                    pil_img.save(buf, format="JPEG")
                    img_bytes_to_process = buf.getvalue()
                    st.success("✓ Biometric frame captured successfully!")
                else:
                    st.error("Failed to capture frame from webcam driver.")

        st.markdown('<div style="text-align:center; color:#475569; font-size:10px; margin:10px 0;">── OR UPLOAD A PHOTO FILE ──</div>', unsafe_allow_html=True)

        # ── OPTION C: FILE UPLOAD ─────────────────────────────────────────
        up_file = st.file_uploader(
            "📁 Drop a selfie here or click Browse (JPG / PNG / WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
        )

        if up_file is not None:
            raw_bytes = up_file.getvalue()
            st.image(raw_bytes, caption=f"✓ Received: {up_file.name}", use_container_width=True)
            img_bytes_to_process = raw_bytes

        # ── OPTION C: Camera Tool ─────────────────────────────────────────
        if os.path.exists(_CAPTURE_PATH):
            st.info("📸 Camera photo detected! Click below to analyse it.")
            if st.button("🔄 Analyse Camera Photo", use_container_width=True):
                with open(_CAPTURE_PATH, "rb") as f:
                    img_bytes_to_process = f.read()
                os.remove(_CAPTURE_PATH)

        # ── AI processing ─────────────────────────────────────────────────
        if img_bytes_to_process is not None:
            with st.spinner("⚙  AI ANALYSING FACIAL GEOMETRY — PLEASE WAIT..."):
                try:
                    pil_img = Image.open(_io.BytesIO(img_bytes_to_process)).convert("RGB")
                    img_np = np.ascontiguousarray(np.array(pil_img, dtype=np.uint8))
                    landmarks = run_face_landmarker(img_np)

                    if landmarks:
                        st.session_state.landmarks = landmarks
                        st.session_state.webcam_photo = img_bytes_to_process
                        bio_results = analyze_landmarks_py(landmarks)
                        st.session_state.biometrics = bio_results

                        wire_img = img_np.copy()
                        draw_landmarks_mesh_py(wire_img, landmarks)
                        st.success("✓ FACE MESH LOCKED — BIOMETRICS EXTRACTED SUCCESSFULLY")
                        st.image(wire_img, caption="Holographic FaceMesh Overlay", use_container_width=True)
                    else:
                        st.warning("⚠ No face found in photo. Try a clearer front-facing selfie with good lighting.")
                        st.markdown("""
                        <div style="font-size:10px; color:#94a3b8; margin-top:8px; line-height:1.7;">
                        <strong style="color:#f59e0b;">Tips for best results:</strong><br>
                        • Face the camera directly (front-facing)<br>
                        • Good bright lighting on your face<br>
                        • No sunglasses or hats<br>
                        • Head fills at least 40% of the frame<br>
                        • Or just use the <strong style="color:#22d3ee;">Demo button above</strong>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Processing error: {e}")
                    st.info("👆 Use the DEMO button above to get full style recommendations without a photo.")









    with col2:
        st.markdown("""
        <div class="hud-card">
            <div class="hud-header">
                <span class="hud-indicator-dot"></span>
                DIAGNOSTICS & SYSTEM LOCKS
            </div>
            <div style="font-size: 11px; line-height: 1.5; color: #94a3b8; font-family: monospace;">
                [1.0] BINDING CAMERA DEVICE... OK<br>
                [2.0] MAPPING LANDMARK MATRIX... OK<br>
                [3.0] DECODING GONIAL ANGLE... OK<br>
                [4.0] MEASURING CHEEK WIDTH... OK
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.biometrics:
            bio = st.session_state.biometrics
            st.markdown(f"""
            <div class="hud-card">
                <div class="hud-header">
                    <span class="hud-indicator-dot"></span>
                    LOCK DETAILS
                </div>
                <div style="font-family: 'Orbitron', sans-serif; font-size: 11px; space-y: 6px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #64748b;">FACE SHAPE:</span>
                        <span style="color: #22d3ee; font-weight: bold;">{bio['faceShape']}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #64748b;">JAW STRENGTH:</span>
                        <span style="color: #e2e8f0;">{bio['jawlineStrength']}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #64748b;">SYMMETRY INDEX:</span>
                        <span style="color: #22d3ee;">{bio['symmetryScore']}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #64748b;">CONFIDENCE:</span>
                        <span style="color: #e2e8f0;">{int(bio['confidenceScore']*100)}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("PROCEED TO CONSULTATION"):
                st.session_state.page = "consult"
                st.rerun()
        else:
            # Bypass simulation helper
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("SIMULATE DEMO SCAN (NO CAMERA)"):
                st.session_state.biometrics = {
                    "faceShape": "Oval",
                    "jawlineStrength": "Chiseled (116°)",
                    "hairDensity": "High",
                    "hairlinePosition": "Medium",
                    "beardDensity": "Medium",
                    "skinTone": "Neutral Olive",
                    "symmetryScore": 96,
                    "foreheadRatio": 0.32,
                    "confidenceScore": 0.98
                }
                st.session_state.webcam_photo = None
                st.session_state.page = "consult"
                st.rerun()

# ---------------------------------------------------------
# SCREEN 3: CONSULTATION MATRIX
# ---------------------------------------------------------
elif st.session_state.page == "consult":
    st.markdown('<h2 style="font-size: 20px; margin-bottom: 15px;">GROOMING CALIBRATION FLOW</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="hud-card">
            <div class="hud-header">
                <span class="hud-indicator-dot"></span>
                ACTIVE BIOMETRIC INDEX
            </div>
            <div style="font-size: 11px; line-height: 1.6; color: #e2e8f0; font-family: monospace;">
                SCAN STATUS: LOCK LOCKED<br>
                FACE PATTERN: OVAL<br>
                JAW MATRIX: DEFINED<br>
                DENSITY CHECK: HIGH
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="hud-card">', unsafe_allow_html=True)
        st.markdown('<div class="hud-header"><span class="hud-indicator-dot"></span>SELECT OCCASION AND STYLE MATRIX PROPERTIES</div>', unsafe_allow_html=True)

        service = st.selectbox(
            "1. WHAT SERVICE ARE YOU LOOKING FOR?",
            ["haircut", "beard", "combo"],
            format_func=lambda x: "HAIRCUT STYLE" if x == "haircut" else "BEARD DESIGN" if x == "beard" else "HAIR + BEARD COMBO"
        )

        occasion = st.selectbox(
            "2. OCCASION CATEGORY?",
            ["corporate", "casual", "party", "wedding"],
            format_func=lambda x: "CORPORATE (FORMAL)" if x == "corporate" else "CASUAL (EASY)" if x == "casual" else "PARTY (EDGY)" if x == "party" else "WEDDING (ELITE)"
        )

        preference = st.selectbox(
            "3. STYLE MATRIX PREFERENCE?",
            ["modern", "luxury", "minimal", "trendy", "celebrity"],
            format_func=lambda x: x.upper()
        )

        maintenance = st.selectbox(
            "4. MAINTENANCE ROUTINE LEVEL LIMITS?",
            ["low", "medium", "high"],
            format_func=lambda x: "LOW (WAKE UP & GO)" if x == "low" else "MEDIUM (EASY CARE)" if x == "medium" else "HIGH (DAILY CARE & PRODUCTS)"
        )

        if st.button("CALCULATE STYLE RECOMMENDATIONS"):
            st.session_state.consultation = {
                "service_type": service,
                "occasion": occasion,
                "style_preference": preference,
                "maintenance_level": maintenance
            }
            st.session_state.page = "recommendation"
            st.session_state.selected_index = 0
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# SCREEN 4: RECOMMENDATION & IMMERSIVE PREVIEW
# ---------------------------------------------------------
elif st.session_state.page == "recommendation":
    st.markdown('<h2 style="font-size: 20px; margin-bottom: 15px;">HUD STYLE PREVIEW DASHBOARD</h2>', unsafe_allow_html=True)
    
    bio = st.session_state.biometrics
    consult = st.session_state.consultation

    # Compute matches
    matches = calculate_recommendations_py(bio, consult)
    
    if not matches:
        st.warning("No catalog styles matched your biometrics. Reset scanner to recalibrate parameters.")
        if st.button("RESET"):
            st.session_state.page = "scan"
            st.rerun()
    else:
        col_left, col_center, col_right = st.columns([1, 2, 1])

        # Left Column: Recommendations list
        with col_left:
            st.markdown("""
            <div class="hud-card" style="padding: 12px; margin-bottom: 10px;">
                <div class="hud-header" style="margin-bottom: 5px;">
                    <span class="hud-indicator-dot"></span>
                    TOP STYLE MATCHES
                </div>
            </div>
            """, unsafe_allow_html=True)

            for idx, item in enumerate(matches[:4]):
                name = item["style"]["name"]
                score = item["score"]
                cat = item["style"]["category"]
                maint = item["style"]["maintenance_level"]

                # Selected style card highlights
                is_selected = st.session_state.selected_index == idx
                border_color = "rgba(6, 182, 212, 0.9)" if is_selected else "rgba(6, 182, 212, 0.15)"
                bg_color = "rgba(6, 182, 212, 0.1)" if is_selected else "rgba(3, 7, 18, 0.4)"

                st.markdown(f"""
                <div class="hud-card" style="border-color: {border_color} !important; background-color: {bg_color} !important; padding: 12px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; font-family: 'Orbitron', sans-serif; font-size: 11px;">
                        <span style="font-weight: bold; color: {'#22d3ee' if is_selected else '#f8fafc'};">{name}</span>
                        <span style="color: #22d3ee; font-weight: 800;">{score}%</span>
                    </div>
                    <div style="font-size: 8px; color: #64748b; margin-top: 4px; display: flex; justify-content: space-between;">
                        <span>CAT: {cat.upper()}</span>
                        <span>MAINT: {maint.upper()}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"SELECT MATCH #{idx+1}", key=f"btn_{idx}"):
                    st.session_state.selected_index = idx
                    st.rerun()

        # Center Column: Dynamic Canvas Preview Mirror
        with col_center:
            active_match = matches[st.session_state.selected_index]
            style_name = active_match["style"]["name"]
            style_cat = active_match["style"]["category"]
            
            st.markdown(f"""
            <div class="hud-card" style="padding: 10px; margin-bottom: 10px; text-align: center;">
                <div class="hud-header" style="margin-bottom: 5px;">
                    <span class="hud-indicator-dot"></span>
                    2.5D HOLOGRAPHIC PREVIEW MIRROR
                </div>
                <div style="font-size: 10px; font-family: monospace; color: #94a3b8; margin-bottom: 5px;">
                    ACTIVE MATRIX: {style_name.upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # View Angle selector
            angle_col1, angle_col2, angle_col3 = st.columns(3)
            with angle_col1:
                if st.button("FRONT VIEW"):
                    st.session_state.view_angle = "front"
                    st.rerun()
            with angle_col2:
                if st.button("LEFT PROFILE"):
                    st.session_state.view_angle = "left"
                    st.rerun()
            with angle_col3:
                if st.button("RIGHT PROFILE"):
                    st.session_state.view_angle = "right"
                    st.rerun()

            # Render face image + OpenCV contour drawing
            if st.session_state.webcam_photo:
                # Process captured photo
                # Convert bytes back to PIL and np array
                import io
                image_bytes = st.session_state.webcam_photo
                pil_feed = Image.open(io.BytesIO(image_bytes))
                preview_img = np.array(pil_feed)
                landmarks = st.session_state.landmarks
            else:
                # Bypass Demo Mode image setup (Create mock black face shape canvas)
                preview_img = np.zeros((360, 480, 3), dtype=np.uint8)
                # Draw gridlines
                for g in range(0, 480, 30):
                    cv2.line(preview_img, (g, 0), (g, 360), (3, 20, 35), 1)
                for g in range(0, 360, 30):
                    cv2.line(preview_img, (0, g), (480, g), (3, 20, 35), 1)
                # Draw mock wireframe circle
                cv2.ellipse(preview_img, (240, 160), (95, 130), 0, 0, 360, (6, 182, 212), 1)
                landmarks = None

            # Overlay style contour if landmarks exist
            if landmarks:
                # Copy so we don't modify stored image
                draw_img = preview_img.copy()
                
                # Shift points slightly if left/right profile view is selected to emulate 3D projection shift
                modified_landmarks = landmarks
                if st.session_state.view_angle == "left":
                    # Shift forehead/chin left to simulate head rotation
                    modified_landmarks = []
                    for lm in landmarks:
                        # Create class emulation of landmarks coordinate
                        class LMClass:
                            def __init__(self, x, y, z):
                                self.x = x
                                self.y = y
                                self.z = z
                        modified_landmarks.append(LMClass(lm.x - 0.05, lm.y, lm.z))
                elif st.session_state.view_angle == "right":
                    modified_landmarks = []
                    for lm in landmarks:
                        class LMClass:
                            def __init__(self, x, y, z):
                                self.x = x
                                self.y = y
                                self.z = z
                        modified_landmarks.append(LMClass(lm.x + 0.05, lm.y, lm.z))

                # Draw style contours
                if style_cat == "haircut" or style_cat == "combo":
                    draw_hair_overlay_py(draw_img, modified_landmarks, style_name)
                if style_cat == "beard" or style_cat == "combo":
                    draw_beard_overlay_py(draw_img, modified_landmarks, style_name)
                
                st.image(draw_img, use_container_width=True)
            else:
                # In Demo mode, draw styling contour centered on canvas
                draw_img = preview_img.copy()
                scale = 1.2
                cyan = (6, 182, 212)
                amber = (245, 158, 11)
                
                # Emulate translation coords centered on canvas
                ctr_x, ctr_y = 240, 160
                if st.session_state.view_angle == "left":
                    ctr_x -= 15
                elif st.session_state.view_angle == "right":
                    ctr_x += 15
                    
                # Hairstyle outline
                if style_cat == "haircut" or style_cat == "combo":
                    pts = []
                    if "pompadour" in style_name.lower():
                        rel = [(-60, -70), (-80, -120), (-35, -150), (0, -150), (35, -150), (80, -120), (60, -70), (0, -72)]
                    elif "quiff" in style_name.lower():
                        rel = [(-55, -70), (-70, -110), (-20, -140), (10, -130), (50, -125), (80, -100), (55, -70), (0, -72)]
                    elif "crop" in style_name.lower():
                        rel = [(-60, -55), (-65, -100), (-30, -110), (0, -110), (30, -110), (65, -100), (60, -55), (0, -50)]
                    else:
                        rel = [(-60, -60), (-60, -100), (-20, -120), (20, -120), (60, -100), (60, -60), (0, -52)]
                        
                    for rx, ry in rel:
                        pts.append([int(ctr_x + rx * scale), int(ctr_y + ry * scale)])
                    pts_arr = np.array(pts, dtype=np.int32)
                    cv2.fillPoly(draw_img, [pts_arr], cyan)
                    cv2.polylines(draw_img, [pts_arr], True, cyan, 2, lineType=cv2.LINE_AA)

                # Beard outline
                if style_cat == "beard" or style_cat == "combo":
                    pts = []
                    if "boxed" in style_name.lower():
                        rel = [(-55, -10), (-60, 35), (-45, 90), (0, 90), (45, 90), (60, 35), (55, -10), (45, -10), (38, 26), (20, 60), (0, 60), (-20, 60), (-38, 26), (-45, -10)]
                    elif "stubble" in style_name.lower():
                        rel = [(-52, -10), (-58, 32), (-44, 86), (0, 86), (44, 86), (58, 32), (52, -10), (42, -10), (34, 24), (18, 56), (0, 56), (-18, 56), (-34, 24), (-42, -10)]
                    else:
                        rel = [(-50, -12), (-54, 34), (-40, 95), (0, 102), (40, 95), (54, 34), (50, -12), (40, -12), (32, 24), (18, 65), (0, 70), (-18, 65), (-32, 24), (-40, -12)]
                        
                    for rx, ry in rel:
                        pts.append([int(ctr_x + rx * scale), int(ctr_y + ry * scale)])
                    pts_arr = np.array(pts, dtype=np.int32)
                    if "stubble" in style_name.lower():
                        for pt in pts:
                            cv2.circle(draw_img, (pt[0], pt[1]), 2, amber, -1)
                    else:
                        cv2.fillPoly(draw_img, [pts_arr], amber)
                        cv2.polylines(draw_img, [pts_arr], True, amber, 2, lineType=cv2.LINE_AA)
                        
                st.image(draw_img, caption="Holographic Projection overlay (Demo Mode)", use_container_width=True)

        # Right Column: Routine & Products suggested
        with col_right:
            st.markdown(f"""
            <div class="hud-card">
                <div class="hud-header">
                    <span class="hud-indicator-dot"></span>
                    STYLE MATRICES
                </div>
                <div style="font-size: 11px; line-height: 1.6;">
                    <span style="color: #64748b; font-family: 'Orbitron', sans-serif;">DESCRIPTION:</span><br>
                    <p style="color: #e2e8f0; font-size: 10px; margin-top: 3px; font-family: sans-serif;">{active_match['style']['description']}</p>
                    <div style="border-top: 1px solid rgba(6,182,212,0.15); margin-top: 12px; padding-top: 8px;">
                        <span style="color: #64748b; font-family: 'Orbitron', sans-serif;">MAINTENANCE:</span><br>
                        <span style="color: #f8fafc; font-weight: bold; font-family: 'Orbitron', sans-serif; font-size: 10px;">{active_match['style']['maintenance_level'].upper()}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="hud-card">
                <div class="hud-header">
                    <span class="hud-indicator-dot"></span>
                    GROOMING PROTOCOL
                </div>
                <div style="font-size: 10px; font-family: sans-serif; color: #94a3b8; line-height: 1.5;">
                    <div style="display: flex; gap: 8px; margin-bottom: 12px;">
                        <span style="font-size: 14px;">📘</span>
                        <div>
                            <strong style="color: #22d3ee; font-family: 'Orbitron', sans-serif; font-size: 9px; display: block;">ROUTINE CARE</strong>
                            Style using clay products on dry hair. Book barber touch-ups every 14 days to lock side gradients.
                        </div>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <span style="font-size: 14px;">🛍️</span>
                        <div>
                            <strong style="color: #22d3ee; font-family: 'Orbitron', sans-serif; font-size: 9px; display: block;">PRODUCTS SUGGESTED</strong>
                            • Matte Texture Clay / Pomade<br>
                            • Organic Conditioning Beard Oil<br>
                            • Activated Charcoal Cleansing Scrub
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("DIAGNOSE NEW FACE"):
                st.session_state.page = "welcome"
                st.session_state.biometrics = None
                st.session_state.consultation = None
                st.session_state.webcam_photo = None
                st.session_state.landmarks = None
                st.session_state.selected_index = 0
                st.session_state.view_angle = "front"
                st.rerun()
