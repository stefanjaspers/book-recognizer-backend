import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.services.visualization_service import VisualizationService

output_dir = "test_output"
os.makedirs(output_dir, exist_ok=True)
vis_service = VisualizationService()


def test_show_mask():
    mask = torch.rand((10, 10))

    # Test with random_color=True
    fig, ax = plt.subplots()
    vis_service.show_mask(mask, ax, random_color=True)
    plt.close(fig)

    # Test with random_color=False (to cover the else statement)
    fig, ax = plt.subplots()
    vis_service.show_mask(mask, ax, random_color=False)
    plt.close(fig)


def test_show_box():
    box = torch.tensor([10, 20, 30, 40])
    label = "Test Label"
    fig, ax = plt.subplots()
    vis_service.show_box(box, ax, label)
    plt.close(fig)


def test_save_mask_data():
    mask_list = torch.rand((3, 10, 10))
    box_list = torch.tensor([[10, 20, 30, 40], [40, 50, 60, 70], [70, 80, 90, 100]])
    label_list = ["Label 1(0.9)", "Label 2(0.8)", "Label 3(0.7)"]

    vis_service.save_mask_data(output_dir, mask_list, box_list, label_list)

    assert os.path.exists(os.path.join(output_dir, "mask.jpg"))
    assert os.path.exists(os.path.join(output_dir, "mask.json"))

    with open(os.path.join(output_dir, "mask.json"), "r") as f:
        json_data = json.load(f)

    assert len(json_data) == 4
    assert json_data[0]["label"] == "background"


def test_draw_output_image():
    # Construct the image path relative to the test file's location
    test_file_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(test_file_dir, "..", "assets", "book-test-2.jpg")

    # Check if the image file exists
    assert os.path.exists(image_path), f"Image file not found: {image_path}"

    masks = torch.rand((3, 10, 10))
    boxes_filt = torch.tensor([[10, 20, 30, 40], [40, 50, 60, 70], [70, 80, 90, 100]])
    pred_phrases = ["Label 1(0.9)", "Label 2(0.8)", "Label 3(0.7)"]

    vis_service.draw_output_image(
        image_path, masks, boxes_filt, pred_phrases, output_dir
    )

    assert os.path.exists(os.path.join(output_dir, "grounded_sam_output.jpg"))
