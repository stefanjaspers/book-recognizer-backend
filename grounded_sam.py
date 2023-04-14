import os

import numpy as np
import json
import torch

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import easyocr

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import build_sam, SamPredictor

# Reads image from given file path, converts it to RGB mode,
# applies a series of transformations (resizing, tensor conversion, normalization),
# and returns both the original PIL image and the transformed image tensor.
def load_image(image_path):
    # Open image and convert to RGB.
    image_pil = Image.open(image_path).convert("RGB")

    transform = T.Compose(
        [
            # Resize shorter side of the image to 800px while keeping aspect ratio intact.
            # max_size ensures the longer side of the resized image does not exceed 1333px.
            T.RandomResize([800], max_size=1333),
            # Convert image to PyTorch tensor with values in the range [0,1].
            T.ToTensor(),
            # Normalizes the image tensor by subtracting the mean values and dividing by
            # the standard deviations for the RGB channels respectively.
            # This normalization is commonly used for pre-trained models from Torchvision.
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Applies the above transformations to the loaded PIL image.
    image, _ = transform(image_pil, None)  # 3, H, W.
    
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    
    _ = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
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
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


# def save_mask_data(output_dir, mask_list, box_list, label_list):
#     value = 0  # 0 for background

#     # mask_img = torch.zeros(mask_list.shape[-2:])
    
#     for idx, mask in enumerate(mask_list):
#         # Create an empty image with the same shape as the mask
#         mask_img = torch.zeros(mask_list.shape[-2:])
        
#         # Set the mask values
#         mask_img[mask.cpu().numpy()[0] == True] = 1

#         # Save the mask image
#         plt.figure(figsize=(10, 10))
#         plt.imshow(mask_img.numpy())
#         plt.axis('off')
#         plt.savefig(os.path.join(output_dir, f'mask{idx}.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

#         # Close the figure to prevent memory leaks
#         plt.close()

#     json_data = [{
#         'value': value,
#         'label': 'background'
#     }]
    
#     for label, box in zip(label_list, box_list):
#         value += 1
#         name, logit = label.split('(')
#         logit = logit[:-1] # the last is ')'
#         json_data.append({
#             'value': value,
#             'label': name,
#             'logit': float(logit),
#             'box': box.numpy().tolist(),
#         })
    
#     with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
#         json.dump(json_data, f)

# def save_mask_data(output_dir, mask_list, image, box_list, label_list):
#     image_np = np.array(image)
    
#     for idx, mask in enumerate(mask_list):
#         mask_np = mask.cpu().numpy().astype(bool)

#         # Remove unnecessary first dimension
#         mask_np = np.squeeze(mask_np, axis=0)

#         # Expand mask along the third axis (channel axis)
#         mask_3d = np.expand_dims(mask_np, axis=2)

#         # Repeat the mask along the channel axis to match the input image shape
#         mask_3d = np.repeat(mask_3d, image_np.shape[2], axis=2)

#         # Extract the object segment using the mask
#         segment = image_np * mask_3d

#         # Convert the NumPy array to a PIL image
#         segment_image = Image.fromarray(segment.astype(np.uint8))

#         # Save the PIL image as a JPEG file
#         segment_image.save(os.path.join(output_dir, f'segment{idx}.jpg'))

# def extract_text_from_segments(mask_list, image, box_list, label_list):
#     image_np = np.array(image)
    
#     extracted_texts = []

#     for idx, mask in enumerate(mask_list):
#         mask_np = mask.cpu().numpy().astype(bool)

#         # Remove unnecessary first dimension
#         mask_np = np.squeeze(mask_np, axis=0)

#         # Expand mask along the third axis (channel axis)
#         mask_3d = np.expand_dims(mask_np, axis=2)

#         # Repeat the mask along the channel axis to match the input image shape
#         mask_3d = np.repeat(mask_3d, image_np.shape[2], axis=2)

#         # Extract the object segment using the mask
#         segment = image_np * mask_3d

#         # Convert the NumPy array to a PIL image
#         segment_image = Image.fromarray(segment.astype(np.uint8))

#         # Run OCR on the extracted segment
#         extracted_text = pytesseract.image_to_string(segment_image)

#         # Add the extracted text to the list
#         extracted_texts.append(extracted_text)
    
#     return extracted_texts

def extract_text_from_segments(mask_list, image, box_list, label_list):
    image_np = np.array(image)
    
    extracted_texts = []

    # Create an EasyOCR reader instance (specify the language as needed)
    reader = easyocr.Reader(['en'])

    for idx, mask in enumerate(mask_list):
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
        segment_image = Image.fromarray(segment.astype(np.uint8))

        # Convert the PIL image to a NumPy array
        segment_np = np.array(segment_image)

        # Run OCR on the extracted segment using EasyOCR
        results = reader.readtext(segment_np)

        # Extract text from the results
        extracted_text = ' '.join([result[1] for result in results])

        # Add the extracted text to the list
        extracted_texts.append(extracted_text)
    
    return extracted_texts
    
# Configuration.
config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
grounded_checkpoint = "checkpoints/groundingdino_swint_ogc.pth"
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
image_path = "assets/book-test-1.png"
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
    point_coords = None,
    point_labels = None,
    boxes = transformed_boxes,
    multimask_output = False,
)
    
# Draw output image.
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
for box, label in zip(boxes_filt, pred_phrases):
    show_box(box.numpy(), plt.gca(), label)

plt.axis('off')
plt.savefig(
    os.path.join(output_dir, "grounded_sam_output.jpg"), 
    bbox_inches="tight", dpi=300, pad_inches=0.0
)

extracted_texts = extract_text_from_segments(masks, image, boxes_filt, pred_phrases)
for text in extracted_texts:
    print(text)