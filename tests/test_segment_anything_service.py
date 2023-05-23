import sys
import os
import pytest
import torch
import cv2
from PIL import Image
from mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.services.segment_anything_service import SegmentAnythingService


@pytest.fixture
def segment_anything_service():
    return SegmentAnythingService()


def test_get_sam_output(segment_anything_service):
    # Mocking the required objects and functions
    sam_checkpoint = "path/to/checkpoint"
    image_path = "path/to/image"
    image_pil = Image.new("RGB", (100, 100))
    boxes_filt = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    with patch(
        "api.services.segment_anything_service.build_sam"
    ) as mock_build_sam, patch(
        "api.services.segment_anything_service.SamPredictor"
    ) as mock_sam_predictor, patch(
        "cv2.imread"
    ) as mock_imread, patch(
        "cv2.cvtColor"
    ) as mock_cvtColor:
        # Set up the return values for the mocked functions
        mock_imread.return_value = MagicMock()
        mock_cvtColor.return_value = MagicMock()
        mock_sam_predictor_instance = MagicMock()
        mock_sam_predictor.return_value = mock_sam_predictor_instance

        # Mock the predict_torch method and set a return value
        mock_masks = torch.tensor([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
        mock_sam_predictor_instance.predict_torch.return_value = (
            mock_masks,
            None,
            None,
        )

        # Call the function to be tested
        result = segment_anything_service.get_sam_output(
            sam_checkpoint, image_path, image_pil, boxes_filt
        )

        # Check if the mocked functions were called with the expected arguments
        mock_build_sam.assert_called_once_with(checkpoint=sam_checkpoint)
        mock_sam_predictor.assert_called_once_with(mock_build_sam.return_value)
        mock_imread.assert_called_once_with(image_path)
        mock_cvtColor.assert_called_once_with(
            mock_imread.return_value, cv2.COLOR_BGR2RGB
        )

        # Add more assertions to check the correctness of the output
        # For example, check if the result is a torch tensor
        assert isinstance(result, torch.Tensor)
