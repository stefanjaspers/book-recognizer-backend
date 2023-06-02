# Stage 1: Download checkpoint files
FROM debian:buster-slim as checkpoints_downloader

# Install curl
RUN apt-get update && apt-get install -y curl

# Create checkpoints directory
RUN mkdir /checkpoints

# Download checkpoint files
RUN curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o /checkpoints/sam_vit_h_4b8939.pth
RUN curl -L https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -o /checkpoints/groundingdino_swint_ogc.pth

# Stage 2: Build the final image
FROM python:3.10.6

WORKDIR /code

COPY requirements.txt .

RUN python3 -m venv .venv
RUN . .venv/bin/activate && pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . .

RUN cd GroundingDINO && pip install -e .
RUN cd ..

COPY --from=checkpoints_downloader /checkpoints /code/checkpoints

COPY .aws /root/.aws

EXPOSE 80

CMD [".venv/bin/uvicorn", "main:app"]