# Use an official PyTorch image with CUDA and cuDNN support for GPU
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /workdir

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# (Optional) Install additional OS packages if needed (for example, unzip)
RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire project code into the container
COPY . .

# Expose ports (JupyterLab on 8888, MLflow on 5000)
EXPOSE 8888 5000

CMD ["tail", "-f", "/dev/null"]
