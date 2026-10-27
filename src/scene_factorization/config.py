import torch

from common.paths import PROJECT_ROOT


def resolve_torch_device() -> str:
    """Choose CUDA when PyTorch exposes a usable GPU."""
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "src" / "scene_factorization" / "scene_factorization_output"
MAX_FOREGROUNDS = 12
EXPORT_ROI_MASK_DILATE_ITERATIONS = 1

DEVICE = resolve_torch_device()

MOT17_ROOT = str(PROJECT_ROOT / "MOT17")
MOT17_SPLIT = "train"
MOT17_SEQUENCES = [
    "MOT17-02-FRCNN",  # 1080p / 30 fps sequence
    "MOT17-04-FRCNN",  # crowded 1080p / 30 fps, existing reference sequence
    "MOT17-05-FRCNN",  # lower-resolution 640x480 / 14 fps sequence
    "MOT17-09-FRCNN",  # shorter 1080p / 30 fps sequence
    "MOT17-10-FRCNN",  # 1080p / 30 fps sequence
    "MOT17-11-FRCNN",  # 1080p / 30 fps sequence
    "MOT17-13-FRCNN",  # 1080p / 25 fps sequence
]

ROI_CONTEXT_SCALE = 1.18
ROI_MIN_EDGE = 24
ROI_NMS_THRESHOLD = 0.75
ROI_MIN_MASK_COVERAGE = 0.20
ROI_MIN_MOTION_SCORE = 0.08
ROI_PERSON_ATTACHED_CLASSES = {24, 26, 28}
ROI_VEHICLE_CLASSES = {1, 2, 3, 5, 7}
ROI_VEHICLE_MOTION_THRESHOLD = 0.40
ROI_VEHICLE_PIXEL_MOTION_THRESHOLD = 0.08

DETECTOR_MODEL_NAME = "yolov8s-seg.pt"
DETECTOR_MODEL_FILE = PROJECT_ROOT / "src" / "common" / "models" / DETECTOR_MODEL_NAME
DETECTOR_MODEL_PATH = str(DETECTOR_MODEL_FILE) if DETECTOR_MODEL_FILE.exists() else DETECTOR_MODEL_NAME
DETECTOR_DEVICE = DEVICE
DETECTOR_CONF = 0.25
DETECTOR_CLASS_CONF = {
    24: 0.10,
    26: 0.10,
    28: 0.10,
}
DETECTOR_ALLOWED_CLASSES = {0, 1, 2, 3, 5, 7, 24, 26, 28}
DETECTOR_MIN_BOX_AREA = 144

TRACKER_MAX_AGE = 40
TRACKER_N_INIT = 3
TRACKER_MAX_IOU_DISTANCE = 0.7
TRACKER_MAX_APPEARANCE_DISTANCE = 0.45
TRACKER_APPEARANCE_WEIGHT = 0.6
TRACKER_IOU_WEIGHT = 0.4
TRACKER_FEATURE_HISTORY = 30
TRACKER_MOTION_EMA_ALPHA = 0.40
TRACKER_MOTION_JITTER_PIXELS = 1.5
TRACKER_MOTION_JITTER_SCALE = 0.015
TRACKER_MOTION_GAIN = 4.0

BACKGROUND_DIFF_THRESHOLD = 24
BACKGROUND_LEARNING_RATE = 0.03
BACKGROUND_FAST_LEARNING_RATE = 0.18
BACKGROUND_CONFIDENCE_GROWTH = 0.08
BACKGROUND_MEDIAN_SAMPLE_COUNT = 60
BACKGROUND_MIN_COMPONENT_AREA = 120
BACKGROUND_PERSISTENCE_ALPHA = 0.18
BACKGROUND_PERSISTENCE_THRESHOLD = 0.35

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
