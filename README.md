# ihomer Book Recognizer AI

This README is still under construction.

The ihomer Book Recognizer AI serves as the backend in a graduation project at ihomer for my BSc at Avans University of Applied Sciences.

The following open-source AI libraries are used during development:

[Segment Anything Model](https://github.com/facebookresearch/segment-anything)
[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
[Amazon Rekognition OCR](https://docs.aws.amazon.com/rekognition/latest/dg/text-detection.html)

The [Google Books API](https://developers.google.com/books/docs/overview) was used to retrieve book information based on the OCR outputs.

I would like to thank the developers of these amazing open-source libraries for their hard work and give them credit for making this project possible.

## :hammer_and_wrench: Install

**Note:**

If you have a CUDA environment, please make sure the environment variable `CUDA_HOME` is set. It will be compiled under CPU-only mode if no CUDA available.

**Installation**

Create a virtual environment and install the necessary dependencies.

```bash
pip install -r requirements.txt
```

Install Segment Anything

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Navigate to the GroundingDINO directory.

```bash
cd GroundingDINO
```

Install GroundingDINO using the following command:

```bash
pip install -e .
```

Create a new directory called "checkpoints" to store the latest model weights, which can be downloaded here:

Grounding DINO:

```bash
https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Segment Anything:

```bash
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```