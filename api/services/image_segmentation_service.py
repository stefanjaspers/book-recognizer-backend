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
        print("nparr shape: ", nparr.shape)
        print("nparr dtype: ", nparr.dtype)
        print("nparr sample values: ", nparr[0:10])


        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite('image_imdecode.jpg', image)


        # Convert from BGR to RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('image_BGR2RGB.jpg', image)
        print("image shape: ", image.shape)
        print("image dtype: ", image.dtype)
        print("image sample values: ", image[0:10])


        # Convert the image to a NumPy array
        image_np = np.array(image)
        print("image_np shape: ", image_np.shape)
        print("image_np dtype: ", image_np.dtype)
        print("image_np sample values: ", image_np[0:10])


        extracted_texts = []

        # Iterate through each mask and box pair
        for mask, box in zip(mask_list, box_list):
            # Convert the mask from a PyTorch tensor to a boolean NumPy array
            mask_np = mask.cpu().numpy().astype(bool)

            # Remove unnecessary first dimension
            mask_np = np.squeeze(mask_np, axis=0)

            # Expand mask along the third axis (channel axis)
            mask_3d = np.expand_dims(mask_np, axis=2)
            print("mask_3d shape: ", mask_3d.shape)
            print("mask_3d dtype: ", mask_3d.dtype)
            print("mask_3d sample values: ", mask_3d[0:10])


            # Repeat the mask along the channel axis to match the input image shape
            mask_3d = np.repeat(mask_3d, image_np.shape[2], axis=2)
            print("mask_3d 2 shape: ", mask_3d.shape)
            print("mask_3d 2 dtype: ", mask_3d.dtype)
            print("mask_3d 2 sample values: ", mask_3d[0:10])

            # Extract the object segment using the mask
            segment = image_np * mask_3d
    
            # Convert the NumPy array to a PIL image
            segment_image = Image.fromarray(segment)
            cv2.imwrite('segment_image.jpg', image)

            # Convert the PyTorch tensor to a NumPy array and get the bounding box coordinates
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            print("x1: ", x1)
            print("y1: ", y1)
            print("x2: ", x2)
            print("y2: ", y2)


            # Crop the image using the bounding box
            cropped_segment_image = segment_image.crop((x1, y1, x2, y2))
            cv2.imwrite('cropped_segment_image.jpg', image)

            # Run the OCR model for each cropped book and append the text to the list
            extracted_texts.append(
                aws_rekognition_service.extract_text_from_segment(cropped_segment_image)
            )

        return extracted_texts
