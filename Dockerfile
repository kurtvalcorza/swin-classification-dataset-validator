# Runnable validator worker image.
# Uses the same digest-pinned base as the finetuner worker for runtime parity
# with the qualification evidence (the validator itself needs only Pillow,
# which the package install pins).
FROM pytorch/pytorch@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385

COPY . /opt/worker/src
RUN pip install --no-cache-dir /opt/worker/src

ENTRYPOINT ["swin-classification-validate"]
