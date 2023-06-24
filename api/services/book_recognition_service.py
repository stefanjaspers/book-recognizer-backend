import os
import json
import base64
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Services.
from api.services.grounding_dino_service import GroundingDINOService
from api.services.image_segmentation_service import ImageSegmentationService
from api.services.segment_anything_service import SegmentAnythingService
from api.services.visualization_service import VisualizationService
from api.services.google_books_service import GoogleBooksService

# Initialize services.
grounding_dino_service = GroundingDINOService()
image_segmentation_service = ImageSegmentationService()
segment_anything_service = SegmentAnythingService()
visualization_service = VisualizationService()
google_books_service = GoogleBooksService()

script_dir = os.path.dirname(os.path.realpath(__file__))
config_file_path = os.path.join(script_dir, "..", "config", "ml_config.json")

with open(config_file_path, "r") as f:
    config = json.load(f)


class BookRecognitionService:
    def __init__(self) -> None:
        pass

    async def recognize(self, image_base64):
        image_data = base64.b64decode(image_base64)

        image_buffer = BytesIO(image_data)

        print("Loading image")

        # Load image.
        image_pil, image_tensor = grounding_dino_service.load_image(image_buffer)

        print("Loading model")

        # Load model.
        config_file = os.path.join(script_dir, "..", "..", config["config_file"])
        grounded_checkpoint = os.path.join(
            script_dir, "..", "..", config["grounded_checkpoint"]
        )
        model = grounding_dino_service.load_model(
            config_file, grounded_checkpoint, config["device"]
        )

        print("Running Grounding DINO model")

        # Run Grounding DINO model.
        boxes_filt, pred_phrases = grounding_dino_service.get_grounding_output(
            model,
            image_tensor,
            config["text_prompt"],
            config["box_threshold"],
            config["text_threshold"],
            config["device"],
        )

        # Convert tensor to numpy if it's not already
        if isinstance(boxes_filt, torch.Tensor):
            boxes_filt = boxes_filt.cpu().numpy()

        fig, ax = plt.subplots(1)
        ax.imshow(image_pil)

        # Image dimensions
        width, height = image_pil.size

        for i, box in enumerate(boxes_filt):
            # Convert from center_x, center_y, w, h to x1, y1, x2, y2 and scale
            center_x, center_y, w, h = box
            x1 = (center_x - w / 2) * width
            y1 = (center_y - h / 2) * height
            x2 = (center_x + w / 2) * width
            y2 = (center_y + h / 2) * height

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Add the predicted phrase as a label
            ax.text(x1, y1, pred_phrases[i], color='r')

        plt.show()



        print("Running SAM model")

        # Run Segment Anything Model.
        sam_checkpoint = os.path.join(script_dir, "..", "..", config["sam_checkpoint"])
        masks = segment_anything_service.get_sam_output(
            sam_checkpoint, image_buffer, image_pil, boxes_filt
        )

        to_pil = T.ToPILImage()

        # Make sure to normalize tensor to [0, 1] range if it's not already done
        for i, tensor in enumerate(masks):
            # Convert tensor to PIL image
            image = to_pil(tensor.float())

            # Save the PIL image
            image.save(f'image_{i}.jpg')

        print("Running OCR model")

        # Run Rekognition OCR model and store book texts in list.
        book_texts = image_segmentation_service.segment_books(
            masks, image_buffer, boxes_filt
        )

        print("Processing texts")

        processed_book_texts = google_books_service.process_book_texts(book_texts)

        url_encoded_book_texts = google_books_service.url_encode_strings(
            processed_book_texts
        )

        api_urls = google_books_service.create_api_urls(url_encoded_book_texts)

        book_list = await google_books_service.get_book_list(api_urls)

        return book_list
