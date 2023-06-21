import numpy as np
import cv2
from PIL import Image

# Services
from api.services.aws_rekognition_service import AWSRekognitionService

# Initialize service
aws_rekognition_service = AWSRekognitionService()


class ImageSegmentationService:
    def __init__(self) -> None:
        pass

    def segment_books(self, mask_list, image, box_list):
        image_bytes = image.getvalue()

        nparr = np.frombuffer(image_bytes, np.uint8)

        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert from BGR to RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a NumPy array
        image_np = np.array(image)

        extracted_texts = []

        # Iterate through each mask and box pair
        for mask, box in zip(mask_list, box_list):
            # Convert the mask from a PyTorch tensor to a boolean NumPy array
            mask_np = mask.cpu().numpy().astype(bool)

            # Remove unnecessary first dimension
            mask_np = np.squeeze(mask_np, axis=0)

            # Expand mask along the third axis (channel axis)
            mask_3d = np.expand_dims(mask_np, axis=2)

            # Repeat the mask along the channel axis to match the input image shape
            mask_3d = np.repeat(mask_3d, image_np.shape[2], axis=2)

            # Extract the object segment using the mask
            segment = image_np * mask_3d

            # Convert the NumPy array to a PIL image
            segment_image = Image.fromarray(segment)

            # Convert the PyTorch tensor to a NumPy array and get the bounding box coordinates
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)

            # Crop the image using the bounding box
            cropped_segment_image = segment_image.crop((x1, y1, x2, y2))

            # Run the OCR model for each cropped book and append the text to the list
            extracted_texts.append(
                aws_rekognition_service.extract_text_from_segment(cropped_segment_image)
            )

        return extracted_texts
