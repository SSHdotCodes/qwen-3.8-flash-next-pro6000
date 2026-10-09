FROM lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469

# SGLang v0.5.21 with the tested dependency repair. Keep CUDA 13 NIXL;
# remove the unused generic CUDA 12 shim and optional Cosmos guardrail.
RUN python3 -m pip uninstall -y nixl cosmos-guardrail && \
    python3 -m pip install \
      nvidia-nccl-cu13==2.29.7 protobuf==6.33.6 grpcio-tools==1.81.1 \
      typeguard==4.6.0 pillow==11.3.0 opencv-python==5.0.0.93 && \
    python3 -m pip check

COPY overlays/sglang/ /sgl-workspace/sglang/python/sglang/
COPY overlays/hot-token-ids.json /opt/qwen-mtp-hotmap.json

# Decode kernels (sglang/srt/qwenfast.py loads /opt/qwenfast/qwenfast.so): compiled here against the
# image's PyTorch and CUDA 13.0 toolkit for SM120a; no GPU is needed to build.
COPY qwenfast/ /opt/qwenfast/src/
ARG MAX_JOBS=8
RUN cd /opt/qwenfast/src && MAX_JOBS=$MAX_JOBS python3 build_qf.py && \
    cp build_qf/qwenfast.so /opt/qwenfast/qwenfast.so
COPY runtime.json verify-runtime.py /opt/flash-next/
RUN python3 /opt/flash-next/verify-runtime.py --installed

ARG RUNTIME_MANIFEST_SHA256
LABEL org.opencontainers.image.source="https://github.com/SSHdotCodes/qwen-3.8-flash-next-pro6000" \
      local.flash-next.release="0.5.21-pro6000-20261009-a16" \
      local.flash-next.manifest-sha256="${RUNTIME_MANIFEST_SHA256}"
