FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive             PIP_NO_CACHE_DIR=1             PYTHONUNBUFFERED=1             AO_GRASP_ROOT=/opt/ao-grasp

RUN apt-get update && apt-get install -y --no-install-recommends             build-essential             ca-certificates             git             python3             python3-dev             python3-pip             && ln -sf /usr/bin/python3 /usr/bin/python             && rm -rf /var/lib/apt/lists/*

COPY docker/aograsp/cgn.requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel             && python -m pip install -r /tmp/requirements.txt

COPY third_party/ao-grasp ${AO_GRASP_ROOT}
COPY docker/aograsp/services /opt/service

RUN test -f ${AO_GRASP_ROOT}/contact_graspnet/compile_pointnet_tfops.sh             && test -f ${AO_GRASP_ROOT}/contact_graspnet/contact_graspnet/contact_grasp_estimator.py

WORKDIR ${AO_GRASP_ROOT}/contact_graspnet
RUN bash compile_pointnet_tfops.sh

WORKDIR /opt/service
CMD ["python", "cgn_service.py"]
