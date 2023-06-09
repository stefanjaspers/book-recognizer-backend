import sys
import torch
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

# Grounding DINO.
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)


class GroundingDINOService:
    def __init__(self) -> None:
        pass

    def load_image(self, image_path):
        # Convert BytesIO to numpy array
        image_np = np.frombuffer(image_path.getvalue(), dtype=np.uint8)

        # Decode the image from numpy array
        image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Convert the image from BGR to RGB
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        # Convert the OpenCV image (numpy array) to PIL image
        image_pil = Image.fromarray(image_cv)

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        image, _ = transform(image_pil, None)

        return image_pil, image

    def load_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device

        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")

        _ = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _ = model.eval()

        return model

    def get_grounding_output(
        self,
        model,
        image,
        caption,
        box_threshold,
        text_threshold,
        with_logits=True,
        device="cpu",
    ):
        caption = caption.lower()
        caption = caption.strip()

        if not caption.endswith("."):
            caption = caption + "."

        model = model.to(device)
        image = image.to(device)

        with torch.no_grad():
            outputs = model(image[None], captions=[caption])

        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # Filter output.
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # Get phrase.
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)

        # Build prediction.
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenlizer
            )
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases
