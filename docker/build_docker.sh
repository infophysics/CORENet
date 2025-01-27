NVC_TAG=23.12
NERSC_TAG="ngc-${NVC_TAG}-v0"
IMAGE_LABEL="corenet"

docker build --platform linux/amd64 --build-arg nvc_tag=$NVC_TAG-py3 -t $IMAGE_LABEL/latest:$NERSC_TAG -f Dockerfile .