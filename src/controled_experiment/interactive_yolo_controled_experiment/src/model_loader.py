from ultralytics import YOLOE, SAM, FastSAM
from urllib.request import urlretrieve
from interactive_yolo_utils import workspace_dir
import os

class YOLOEMod(YOLOE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_vpe(
        self,
        visual_prompts: dict,
        refer_image,
        predictor,
        **kwargs,
    ):


        #assert ("bboxes" in visual_prompts or "masks" in visual_prompts) and "cls" in visual_prompts, (
        #    f"Expected 'bboxes' or 'masks' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
        #)
        #assert len(visual_prompts["bboxes"]) == len(visual_prompts["cls"]), (
        #    f"Expected equal number of bounding boxes and classes, but got {len(visual_prompts['bboxes'])} and "
        #    f"{len(visual_prompts['cls'])} respectively"
        #)

        self.predictor = (predictor or self._smart_load("predictor"))(
            overrides={
                "task": self.model.task,
                "mode": "predict",
                "save": False,
                "verbose": refer_image is None,
                "batch": 1,
            },
            _callbacks=self.callbacks,
        )

        num_cls = len(set(visual_prompts["cls"]))
        self.model.model[-1].nc = num_cls
        self.model.names = [f"object{i}" for i in range(num_cls)]
        self.predictor.set_prompts(visual_prompts.copy())

        self.predictor.setup_model(model=self.model)
        
        vpe = self.predictor.get_vpe(refer_image)
        self.predictor = None  # reset predictor
        return vpe

YOLO_MODEL_NAME = "yoloe-11l-seg"

def models_dir():
    dir_path = os.path.join(workspace_dir(), "models")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created models directory: {dir_path}")
    return dir_path

def yoloe_model()->YOLOE:
    base_model_path = os.path.join(models_dir(), YOLO_MODEL_NAME+".pt")
    base_model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11m-seg.pt"

    if not os.path.exists(base_model_path):
        urlretrieve(base_model_url, base_model_path)

    return YOLOEMod(base_model_path)

def sam_model()->SAM:
    sam_model_path = os.path.join(models_dir(), "sam2_l.pt")
    sam_model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_l.pt"

    if not os.path.exists(sam_model_path):
        urlretrieve(sam_model_url, sam_model_path)

    return SAM(sam_model_url)

FAST_SAM_MODEL_NAME = "FastSAM-x"

def fast_sam_model()->FastSAM:
    engine_sam_model_path = os.path.join(models_dir(), FAST_SAM_MODEL_NAME + ".engine")
    sam_model_path = os.path.join(models_dir(), FAST_SAM_MODEL_NAME + ".pt")
    sam_model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-x.pt"

    if not os.path.exists(engine_sam_model_path):

        if not os.path.exists(sam_model_path):
            urlretrieve(sam_model_url, sam_model_path)

        return FastSAM(sam_model_url)
    
    else:
        return FastSAM(engine_sam_model_path)
    
def generate_fast_sam_engine_model(use_dla = False):
    sam_model_path = os.path.join(models_dir(), FAST_SAM_MODEL_NAME + ".pt")
    engine_sam_model_path = os.path.join(models_dir(), FAST_SAM_MODEL_NAME + ".engine")
    engine_sam_model_generation_path = os.path.join("weights", FAST_SAM_MODEL_NAME + ".engine")
    sam_model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-x.pt"

    if not os.path.exists(sam_model_path):
            urlretrieve(sam_model_url, sam_model_path)

    model = FastSAM(sam_model_url)

    if(use_dla):
        model.export(format="engine", batch=1, device="dla:0", dynamic=True)
    else:
        model.export(format="engine", batch=1, device="0", dynamic=True)

    os.replace(engine_sam_model_generation_path, engine_sam_model_path)
