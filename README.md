## Introduction

This is an unofficial replication of "Pix2seq: A Language Modeling Framework for Object Detection" with pretrained model on mmdetection.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

## Train & Evaluation

Train by running (about 10 days with 8*V100 32GB)
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=5003 \
  tools/train.py configs/pix2seq/pix2seq_r50_8x4_50e_coco.py --work-dir pix2seq-output --gpus 8 --launcher pytorch
```

or

Download [pretrained pix2seq weights](https://drive.google.com/file/d/1Ku8ZORiLtMs66uleS3aXId7pxlJrTK9d/view?usp=sharing).

Evaluate with single gpu:
```bash
python tools/test.py configs/pix2seq/pix2seq_r50_8x4_300_coco.py \
  weights/checkpoints.pth --work-dir pix2seq-output --eval bbox --show-dir pix2seq-vis
```

Evaluate with 8 gpus:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=5003 \
  tools/test.py configs/pix2seq/pix2seq_r50_8x4_300_coco.py weights/checkpoints.pth \
  --work-dir pix2seq-output --eval bbox --launcher pytorch
```

| Method  | backbone | Epoch | Batch Size | AP   | AP50  | AP75  | Weights |
| :-----: | :------: | :----:| :---------:| :---:| :---: | :---: | :---:   |
| Ours    | R50      | 300   | 32         | 36.4 | 52.8  | 38.5  | [model](https://github.com/Sharpiless/Pix2seq-mmdetection/releases/download/v1.0/checkpoints.pth) |
| Paper   | R50      | 300   | 128        | 43.0 | 61.0  | 45.6  | - |


## Visualization

![](https://github.com/Sharpiless/mmdet-Pix2Seq/blob/main/resources/007114.jpg)

![](https://github.com/Sharpiless/mmdet-Pix2Seq/blob/main/resources/007351.jpg)

![](https://github.com/Sharpiless/mmdet-Pix2Seq/blob/main/resources/008322.jpg)

![](https://github.com/Sharpiless/mmdet-Pix2Seq/blob/main/resources/000000289393.jpg)

![](https://github.com/Sharpiless/mmdet-Pix2Seq/blob/main/resources/000000212559.jpg)

![](https://github.com/Sharpiless/mmdet-Pix2Seq/blob/main/resources/000000255664.jpg)

## TO-DO

- [x] random shuffle targets
- [x] training from scratch
- [x] drop class token
- [x] stochastic depth
- [x] large scale jittering
- [ ] support for custom dataset
- [x] two independent augmentations for each image
- [x] FrozenBatchNorm2d in backbones
- [x] auto-argument
- [x] nucleus sampling

## Acknowledgement

[https://github.com/gaopengcuhk/Pretrained-Pix2Seq](https://github.com/gaopengcuhk/Pretrained-Pix2Seq)

[https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
