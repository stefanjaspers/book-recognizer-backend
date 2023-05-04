import matplotlib.pyplot as plt
import numpy as np
import os
import json
import torch
import cv2


class VisualizationService:
    def __init__(self) -> None:
        pass

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        ax.imshow(mask_image)

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]

        ax.add_patch(
            plt.Rectangle(
                (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
            )
        )
        ax.text(x0, y0, label)

    def save_mask_data(self, output_dir, mask_list, box_list, label_list):
        value = 0  # 0 for background

        mask_img = torch.zeros(mask_list.shape[-2:])

        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis("off")
        plt.savefig(
            os.path.join(output_dir, "mask.jpg"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.0,
        )

        json_data = [{"value": value, "label": "background"}]

        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split("(")
            logit = logit[:-1]  # the last is ')'
            json_data.append(
                {
                    "value": value,
                    "label": name,
                    "logit": float(logit),
                    "box": box.numpy().tolist(),
                }
            )

        with open(os.path.join(output_dir, "mask.json"), "w") as f:
            json.dump(json_data, f)

    def draw_output_image(
        self, image_path, masks, boxes_filt, pred_phrases, output_dir
    ):
        plt.figure(figsize=(10, 10))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)

        for mask in masks:
            self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)

        for box, label in zip(boxes_filt, pred_phrases):
            self.show_box(box.numpy(), plt.gca(), label)

        plt.axis("off")
        plt.savefig(
            os.path.join(output_dir, "grounded_sam_output.jpg"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.0,
        )
