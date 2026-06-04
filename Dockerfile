# HyperCP reproducibility image.
#
#   docker build -t hypercp .
#   docker run --rm -it hypercp                 # runs the quick CPU smoke
#   docker run --rm -it hypercp python make_all.py --device cuda:0   # full (GPU)
#
# The model is pure PyTorch; no PyTorch Geometric or DGL is needed.
FROM python:3.10-slim

WORKDIR /workspace

# libgomp1 is required by LightGBM's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: regenerate every results JSON with the fast CPU smoke config.
# Drop --quick (and pass --device cuda:0) for the full production run.
CMD ["python", "make_all.py", "--quick"]
