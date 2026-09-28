# syntax=docker/dockerfile:1
# 本番で完全固定する場合は、検証したbase imageのsha256 digestを指定する。
ARG BASE_IMAGE=public.ecr.aws/lambda/python:3.12
FROM ${BASE_IMAGE} AS dependencies

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

RUN dnf install -y libgomp && dnf clean all
COPY requirements.txt constraints.txt /build/
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0+cpu" \
    && python -m pip install -r /build/requirements.txt -c /build/constraints.txt \
    && python -m pip check

FROM dependencies AS model
COPY model.lock.json /build/model.lock.json
COPY build/download_model.py /build/download_model.py
# このステージだけがHugging Faceへアクセスする。runtimeにはcacheを持ち込まない。
RUN python /build/download_model.py --lock /build/model.lock.json --output /opt/laya-model

FROM dependencies AS runtime
COPY --from=model /opt/laya-model /opt/laya-model
COPY src/ ${LAMBDA_TASK_ROOT}/
COPY build/smoke_test.py /opt/build-smoke-test.py

ENV MODEL_DIR=/opt/laya-model \
    HF_HOME=/tmp/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    TORCH_HOME=/tmp/torch \
    XDG_CACHE_HOME=/tmp/cache \
    HOME=/tmp \
    TORCH_NUM_THREADS=4 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    LAYA_MAX_LEN=1024 \
    LAYA_HEAD_MAX_LEN=384 \
    LAYA_MAX_QUESTIONS=8

RUN python -m pip freeze > /opt/python-dependencies.txt \
    && chmod -R a+rX /opt/laya-model ${LAMBDA_TASK_ROOT} \
    && python /opt/build-smoke-test.py \
    && rm -rf /tmp/huggingface /tmp/torch /tmp/cache

CMD ["handler.lambda_handler"]
