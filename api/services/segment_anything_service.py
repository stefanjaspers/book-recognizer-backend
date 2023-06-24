import cv2
import numpy as np
import torch

# Segment Anything.
from segment_anything import build_sam, SamPredictor


class SegmentAnythingService:
    def __init__(self) -> None:
        pass

    def get_sam_output(self, sam_checkpoint, image, image_pil, boxes_filt):
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device="cpu"))

        image_bytes = image.getvalue()

        nparr = np.frombuffer(image_bytes, np.uint8)

        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device="cpu")

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device="cpu"),
            multimask_output = False,
        )

        return masks