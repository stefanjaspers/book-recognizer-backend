import os
import sys
import pytest
from unittest.mock import MagicMock
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.services.aws_rekognition_service import AWSRekognitionService


@pytest.fixture
def aws_rekognition_service():
    return AWSRekognitionService()


def test_extract_text_from_segment(aws_rekognition_service):
    # Mock boto3.client
    aws_rekognition_service.client = MagicMock()

    # Mock response from AWS Rekognition
    mock_response = {
        "TextDetections": [
            {"Type": "LINE", "DetectedText": "Hello"},
            {"Type": "LINE", "DetectedText": "World"},
        ]
    }

    # Mock detect_text method
    aws_rekognition_service.client.detect_text = MagicMock(return_value=mock_response)

    # Create a sample image for testing
    image = Image.new("RGB", (100, 100))

    # Call the extract_text_from_segment method
    result = aws_rekognition_service.extract_text_from_segment(image)

    # Assert the expected result
    assert result == "Hello World"
