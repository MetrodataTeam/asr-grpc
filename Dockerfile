FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG target=/mdt/run

WORKDIR ${target}
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/opt/conda/lib/python3.11/site-packages/torch/lib/:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
RUN apt update && \
  apt install -y --no-install-recommends ffmpeg && \
  pip install --no-cache-dir -i ${PYPI} poetry==2.0.1 && \
  poetry config virtualenvs.create false

ADD asr/pyproject.toml asr/poetry.lock ${target}/

RUN poetry install --no-cache --no-root --with gpu

COPY asr/pb/ ${target}/pb/
COPY asr/*.py ${target}/

RUN python -m compileall ${target}
CMD ["python", "server.py"]
EXPOSE 18909
