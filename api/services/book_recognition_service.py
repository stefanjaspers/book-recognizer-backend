import os
import json

# Services.
from .grounding_dino_service import GroundingDINOService
from .image_segmentation_service import ImageSegmentationService
from .segment_anything_service import SegmentAnythingService
from .visualization_service import VisualizationService

# Initialize services.
grounding_dino_service = GroundingDINOService()
image_segmentation_service = ImageSegmentationService()
segment_anything_service = SegmentAnythingService()
visualization_service = VisualizationService()

script_dir = os.path.dirname(os.path.realpath(__file__))
config_file_path = os.path.join(script_dir, "..", "config", "ml_config.json")

with open(config_file_path, "r") as f:
    config = json.load(f)


class BookRecognitionService:
    def __init__(self) -> None:
        pass

    def recognize(self):
        # Create output directory.
        os.makedirs(config["output_dir"], exist_ok=True)

        # Load image.
        image_path = os.path.join(script_dir, "..", "..", config["image_path"])
        image_pil, image = grounding_dino_service.load_image(image_path)

        # Load model.
        config_file = os.path.join(script_dir, "..", "..", config["config_file"])
        grounded_checkpoint = os.path.join(script_dir, "..", "..", config["grounded_checkpoint"])
        model = grounding_dino_service.load_model(
            config_file, grounded_checkpoint, config["device"]
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
        sam_checkpoint = os.path.join(script_dir, "..", "..", config["sam_checkpoint"])
        masks = segment_anything_service.get_sam_output(
            sam_checkpoint, image_path, image_pil, boxes_filt
        )

        # (optional) Draw output image.
        output_dir = os.path.join(script_dir, "..", "..", config["output_dir"])
        visualization_service.draw_output_image(
            image_path, masks, boxes_filt, pred_phrases, output_dir
        )

        # (optional) Save object masks.
        visualization_service.save_mask_data(
            output_dir, masks, boxes_filt, pred_phrases
        )

        # Run Rekognition OCR model and store book texts in list.
        book_texts = image_segmentation_service.segment_books(
            masks, image_path, boxes_filt
        )

        return book_texts