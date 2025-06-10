import os
from .base_model import YOLO_MODEL_NAME, models_dir
from ultralytics import YOLO, YOLOE

def generate_fast_model(model: YOLOE, img_exemple, use_dla : bool = False) -> None:
    """
    Save the YOLO model in TensorRT format.
    """
    export_path = YOLO_MODEL_NAME + ".torchscript"
    fast_model_path = os.path.join(models_dir(), export_path)

    device = "dla:0" if use_dla else "0"

    model.export(format="torchscript", batch=1, device=device)

    fast_model = YOLO(fast_model_path)
    fast_model(img_exemple)
    os.remove(fast_model_path)
    
    return fast_model