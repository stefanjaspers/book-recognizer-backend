import sys
import os
from PIL import Image
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.services.grounding_dino_service import GroundingDINOService

# Initialize the GroundingDINOService object
gds = GroundingDINOService()

# Test data
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(base_dir, "..")
image_path = os.path.join(project_root, "assets/book-test-2.jpg")
model_config_path = os.path.join(
    project_root, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
model_checkpoint_path = os.path.join(
    project_root, "checkpoints/groundingdino_swint_ogc.pth"
)
device = "cpu"
caption = "A sample caption."
box_threshold = 0.5
text_threshold = 0.5


def test_load_image():
    image_pil, image = gds.load_image(image_path)
    assert isinstance(image_pil, Image.Image)
    assert isinstance(image, torch.Tensor)


def test_load_model():
    model = gds.load_model(model_config_path, model_checkpoint_path, device)
    assert hasattr(model, "eval")


def test_get_grounding_output():
    # Load image and model
    image_pil, image = gds.load_image(image_path)
    model = gds.load_model(model_config_path, model_checkpoint_path, device)

    # Test case 1: Caption without a period at the end
    caption_no_period = "A sample caption"
    boxes_filt, pred_phrases = gds.get_grounding_output(
        model,
        image,
        caption_no_period,
        box_threshold,
        text_threshold,
        with_logits=True,
        device=device,
    )
    assert isinstance(boxes_filt, torch.Tensor)
    assert isinstance(pred_phrases, list)
    assert all(isinstance(phrase, str) for phrase in pred_phrases)

    # Test case 2: Caption with a period at the end and with_logits=False
    caption_with_period = "A sample caption."
    boxes_filt, pred_phrases = gds.get_grounding_output(
        model,
        image,
        caption_with_period,
        box_threshold,
        text_threshold,
        with_logits=False,
        device=device,
    )
    assert isinstance(boxes_filt, torch.Tensor)
    assert isinstance(pred_phrases, list)
    assert all(isinstance(phrase, str) for phrase in pred_phrases)

    # Test case 3: Lower text_threshold to cover the remaining code
    lower_text_threshold = 0.1
    boxes_filt, pred_phrases = gds.get_grounding_output(
        model,
        image,
        caption_with_period,
        box_threshold,
        lower_text_threshold,
        with_logits=True,
        device=device,
    )
    assert isinstance(boxes_filt, torch.Tensor)
    assert isinstance(pred_phrases, list)
    assert all(isinstance(phrase, str) for phrase in pred_phrases)