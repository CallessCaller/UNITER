IMG_DIR=$1
OUT_DIR=$2

set -e

echo "extracting image features..."
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
sudo docker run --gpus '"'device=2'"' --ipc=host --rm \
    --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
    --mount src=$OUT_DIR,dst=/output,type=bind \
    -w /src chenrocks/butd-caffe:nlvr2 \
    bash -c "python tools/generate_npz.py --gpu 0"

echo "done"
