from io import BytesIO

# AWS SDK
import boto3


class AWSRekognitionService:
    def __init__(self):
        self.client = boto3.client("rekognition")

    def extract_text_from_segment(self, image):
        # Create a buffer to hold the binary data
        buffer = BytesIO()

        # Save the PIL image in the buffer using the specified format
        image.save(buffer, format="PNG")

        # Get the binary data from the buffer
        image_bytes = buffer.getvalue()

        response = self.client.detect_text(Image={"Bytes": image_bytes})

        combined_string = ""

        text_detections = response["TextDetections"]

        for text in text_detections:
            if text["Type"] == "LINE":
                combined_string += text["DetectedText"] + " "

        # Remove the extra space at the end of the combined string
        combined_string = combined_string.strip()

        return combined_string
