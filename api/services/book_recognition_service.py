import os
import json

# Services.
from grounding_dino_service import GroundingDINOService
from image_segmentation_service import ImageSegmentationService
from segment_anything_service import SegmentAnythingService
from visualization_service import VisualizationService

# Initialize services.
grounding_dino_service = GroundingDINOService()
image_segmentation_service = ImageSegmentationService()
segment_anything_service = SegmentAnythingService()
visualization_service = VisualizationService()

with open("../config/ml_config.json") as json_file:
    config = json.load(json_file)


class BookRecognitionService:
    def __init__(self) -> None:
        pass

    def recognize():
        # Create output directory.
        os.makedirs(config["output_dir"], exist_ok=True)

        # Load image.
        image_pil, image = grounding_dino_service.load_image(config["image_path"])

        # Load model.
        model = grounding_dino_service.load_model(
            config["config_file"], config["grounded_checkpoint"], config["device"]
        )

        # Run Grounding DINO model.
        boxes_filt, pred_phrases = grounding_dino_service.get_grounding_output(
            model,
            image,
            config["text_prompt"],
            config["box_threshold"],
            config["text_threshold"],
            config["device"],
        )

        # Run Segment Anything Model.
        masks = segment_anything_service.get_sam_output(
            config["sam_checkpoint"], config["image_path"], image_pil, boxes_filt
        )

        # (optional) Draw output image.
        visualization_service.draw_output_image(
            config["image_path"], masks, boxes_filt, pred_phrases, config["output_dir"]
        )

        # (optional) Save object masks.
        visualization_service.save_mask_data(
            config["output_dir"], masks, boxes_filt, pred_phrases
        )

        # Run Rekognition OCR model and store book texts in list.
        book_texts = image_segmentation_service.segment_books(
            masks, config["image_path"], boxes_filt
        )

        return book_texts
