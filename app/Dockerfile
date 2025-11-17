FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ wget curl jq build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/ /app

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

RUN chmod +x /app/start.sh

EXPOSE 8000

CMD ["/app/start.sh"]
