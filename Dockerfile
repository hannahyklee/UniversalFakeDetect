FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG SERVING_PORT=8000
ENV SERVING_PORT=$SERVING_PORT

WORKDIR /

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install the copied in requirements.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the files necessary to fetch, run and serve the model.
ADD models/ /models/
ADD checkpoints/ /checkpoints/
ADD custommodel.py /custommodel.py
ADD server.py /server.py

# Fetch the model and cache it locally.
# RUN python3 custommodel.py --fetch

# Expose the serving port.
EXPOSE $SERVING_PORT

# Run the server to handle inference requests.
CMD python3 -u server.py