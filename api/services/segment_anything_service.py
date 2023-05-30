import cv2
import torch

# Segment Anything.
from segment_anything import build_sam, SamPredictor


class SegmentAnythingService:
    def __init__(self) -> None:
        pass

    def get_sam_output(self, sam_checkpoint, image_path, image_pil, boxes_filt):
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))

        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)

        size = image_pil.size

        H, W = size[1], size[0]

        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()

        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_filt, image.shape[:2]
        )

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        return masks
