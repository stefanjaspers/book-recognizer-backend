# Stage 1: Download checkpoint files
FROM debian:buster-slim as checkpoints_downloader

# Install curl
RUN apt-get update && apt-get install -y curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create checkpoints directory
RUN mkdir /checkpoints

# Download checkpoint files
RUN curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o /checkpoints/sam_vit_h_4b8939.pth
RUN curl -L https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -o /checkpoints/groundingdino_swint_ogc.pth

# Stage 2: Build the final image
FROM python:3.10.6

WORKDIR /code

COPY requirements.txt .

RUN python3 -m venv .venv && \
    . .venv/bin/activate && pip install --no-cache-dir -r requirements.txt && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . .

RUN cd GroundingDINO && pip install -e . && cd ..

COPY --from=checkpoints_downloader /checkpoints /code/checkpoints

CMD [".venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]