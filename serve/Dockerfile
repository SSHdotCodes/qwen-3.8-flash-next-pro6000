FROM lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f

# Match the tested 0.5.20 dependency repair. Keep CUDA 13 NIXL;
# remove the unused generic CUDA 12 shim and optional Cosmos guardrail.
RUN python3 -m pip uninstall -y nixl cosmos-guardrail && \
    python3 -m pip install \
      nvidia-nccl-cu13==2.29.7 protobuf==6.33.6 grpcio-tools==1.81.1 \
      typeguard==4.6.0 pillow==11.3.0 opencv-python==5.0.0.93 && \
    python3 -m pip check

COPY overlays/sglang/ /sgl-workspace/sglang/python/sglang/
COPY overlays/hot-token-ids.json /opt/qwen-mtp-hotmap.json
COPY runtime.json verify-runtime.py /opt/flash-next/
RUN python3 /opt/flash-next/verify-runtime.py --installed

ARG RUNTIME_MANIFEST_SHA256
LABEL org.opencontainers.image.source="https://github.com/SSHdotCodes/qwen-3.8-flash-next-pro6000" \
      local.flash-next.release="0.5.20-pro6000-20260919-hotmap" \
      local.flash-next.manifest-sha256="${RUNTIME_MANIFEST_SHA256}"
