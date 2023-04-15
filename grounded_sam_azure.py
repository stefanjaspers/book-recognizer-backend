import os
import time
from io import BytesIO, BufferedReader

import numpy as np
import torch

import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Azure Cognitive Services
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

# Segment Anything
from segment_anything import build_sam, SamPredictor

'''
Authenticate
Authenticates your credentials and creates a client.
'''
subscription_key = "fedc9106cad6447692c5abffb9fb639e"
endpoint = "https://book-detection.cognitiveservices.azure.com/"

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
'''
END - Authenticate
'''


# Reads image from given file path, converts it to RGB mode,
# applies a series of transformations (resizing, tensor conversion, normalization),
# and returns both the original PIL image and the transformed image tensor
def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image, _ = transform(image_pil, None)

    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device

    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")

    _ = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()

    return model


def get_grounding_output(
    model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"
):
    caption = caption.lower()
    caption = caption.strip()

    if not caption.endswith("."):
        caption = caption + "."

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # Filter output.
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # Get phrase.
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)

    # Build prediction.
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer
        )
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
    ax.text(x0, y0, label)


def save_mask_data(mask_list, image, box_list):
    image_np = np.array(image)

    extracted_texts = []

    for _, (mask, box) in enumerate(zip(mask_list, box_list)):
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

        extracted_texts.append(extract_text_from_segment(cropped_segment_image))

    return extracted_texts

    
def extract_text_from_segment(image):
    # Create a buffer to hold the binary data
    buffer = BytesIO()

    # Save the PIL image in the buffer using the specified format
    image.save(buffer, format="PNG")

    # Get the binary data from the buffer
    image_bytes = buffer.getvalue()

    # Create a new BytesIO object from the bytes data
    bytes_io = BytesIO(image_bytes)

    # Create a BufferedReader object from the BytesIO object
    buffered_reader = BufferedReader(bytes_io)

    read_response = computervision_client.read_in_stream(buffered_reader, raw=True)

    read_operation_location = read_response.headers["Operation-Location"]

    operation_id = read_operation_location.split("/")[-1]

    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status.lower () not in ['notstarted', 'running']:
            break
        print ('Waiting for result...')
        time.sleep(10)

    combined_string = ""

    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                combined_string += line.text + " "
    
    # Remove the extra space at the end of the combined string
    combined_string = combined_string.strip()

    return combined_string


# Configuration.
config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
grounded_checkpoint = "checkpoints/groundingdino_swint_ogc.pth"
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
image_path = "assets/book-test-2.jpg"
text_prompt = "book"
output_dir = "outputs"
box_threshold = 0.35
text_threshold = 0.30
device = "cpu"

# Create output directory.
os.makedirs(output_dir, exist_ok=True)
# Load image.
image_pil, image = load_image(image_path)
# Load model.
model = load_model(config_file, grounded_checkpoint, device=device)

# Run GroundingDINO model.
boxes_filt, pred_phrases = get_grounding_output(
    model, image, text_prompt, box_threshold, text_threshold, device=device
)

# Initialize SAM.
predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

size = image_pil.size
H, W = size[1], size[0]
for i in range(boxes_filt.size(0)):
    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
    boxes_filt[i][2:] += boxes_filt[i][:2]

boxes_filt = boxes_filt.cpu()
transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])

masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

# Draw output image.
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
for box, label in zip(boxes_filt, pred_phrases):
    show_box(box.numpy(), plt.gca(), label)

plt.axis("off")
plt.savefig(
    os.path.join(output_dir, "grounded_sam_output.jpg"),
    bbox_inches="tight",
    dpi=300,
    pad_inches=0.0,
)

book_texts = save_mask_data(masks, image, boxes_filt)
for text in book_texts:
    print(text)