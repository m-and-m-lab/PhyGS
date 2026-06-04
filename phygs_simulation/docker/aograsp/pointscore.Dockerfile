FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive             PIP_NO_CACHE_DIR=1             PYTHONUNBUFFERED=1             AO_GRASP_ROOT=/opt/ao-grasp

RUN apt-get update && apt-get install -y --no-install-recommends             build-essential             ca-certificates             git             python3             python3-dev             python3-pip             && ln -sf /usr/bin/python3 /usr/bin/python             && rm -rf /var/lib/apt/lists/*

COPY docker/aograsp/pointscore.requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel             && python -m pip install -r /tmp/requirements.txt             && python -m pip install                 torch==1.13.1+cu117                 torchvision==0.14.1+cu117                 --extra-index-url https://download.pytorch.org/whl/cu117

COPY third_party/ao-grasp ${AO_GRASP_ROOT}
COPY docker/aograsp/services /opt/service

ENV TORCH_CUDA_ARCH_LIST=8.6+PTX

RUN test -f ${AO_GRASP_ROOT}/aograsp/aograsp_model/conf.pth             && test -f ${AO_GRASP_ROOT}/aograsp/aograsp_model/770-network.pth             && python -m pip install --no-build-isolation ${AO_GRASP_ROOT}/aograsp/models/Pointnet2_PyTorch/pointnet2_ops_lib

WORKDIR /opt/service
CMD ["python", "pointscore_service.py"]
