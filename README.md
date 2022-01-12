# UNITER
UNITER without docker image.
You can use any version of Pytorch and CUDA
(Pytorch 1.9.0 & CUDA 11,10,9)

Only VCR data is supported.

# Quickstart
0. Install requirment.txt (Conda environment is recommanded)
    ```bash
    pip install -r requirment.txt
    ```
1. Feature extraction.
    ```bash
    cd rcnn
    ```
    Go to rcnn directory and change paths in extract_features.py

    Then run
    ```bash
    CUDA_VISIBLE_DEVICES=0 python extract_features.py
    ```

2. 2nd stage pretraining.

    First download VCR data into
    "../vcr/vcr1annots/" and "../vcr/vcr1images/"
    and run
    ```bash
    CUDA_VISIBLE_DEVICES=0 python pretrain_vcr.py --batch_size=64 --accum_steps=4
    ```
    After 2nd pretraining, you will get checkpoints in ckpt directory.

3. VCR Finetuning.
    ```bash
    CUDA_VISIBLE_DEVICES=0 python pretrain_vcr.py --ckpt=ckpt/UNITER_2nd_45000_64_4 --batch_size=16 --accum_steps=5 --train_step=8000
    ```
    After finetuning, you will get checkpoints in ckpt/{CURRENT_TIME}/ dir. (ex. ckpt/1641951775.0276442/UNITER_VCR_8000_16_5_6e-05)

4. VCR Evaludation.
    ```bash
    CUDA_VISIBLE_DEVICES=0 python pretrain_vcr.py --ckpt=ckpt/{CURRENT_TIME}/UNITER_VCR_8000_16_5_6e-05 --data_type=val --config=config/uniter-base_vcr.json
    ```
