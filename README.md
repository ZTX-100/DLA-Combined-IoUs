# Dynamic Label Assignment for Object Detection by Combining Predicted and Anchor IoUs

## Introduction

Label assignment plays a significant role in modern object detection models. Detection models may yield totally different performances with different label assignment strategies. For anchor-based detection models, the IoU threshold between the anchors and their corresponding ground truth bounding boxes is the key element since the positive samples and negative samples are divided by the IoU threshold. Early object detectors simply utilize a fixed threshold for all training samples, while recent detection algorithms focus on adaptive thresholds based on the distribution of the IoUs to the ground truth boxes. In this paper, we introduce a simple and effective approach to perform label assignment dynamically based on the training status with predictions. By introducing the predictions in label assignment, more high-quality samples with higher IoUs to the ground truth objects are selected as the positive samples, which could reduce the discrepancy between the classification scores and the IoU scores, and generate more high-quality boundary boxes. Our approach shows improvements in the performance of the detection models with the adaptive label assignment algorithm and lower bounding box losses for those positive samples, indicating more samples with higher quality predicted boxes are selected as positives. Our paper is available at [link](https://arxiv.org/abs/2201.09396).

## Approach
<div style="color:#0000FF" align="center">
<img src="model.pdf" width="430"/>
</div>


## Installation
The implementation of our algorithm is based on [ATSS](https://github.com/sfzhang15/ATSS). Please check [ATSS](https://github.com/sfzhang15/ATSS) and [INSTALL.md](INSTALL.md) for more installation instructions.


## Inference
The inference command line on coco minival split:

    python tools/test_net.py \
        --config-file configs/dynamic_atss/dynamic_atss_R_50_FPN_1x.yaml \
        MODEL.WEIGHT Dynamic_ATSS_R_50_FPN_1x.pth \
        TEST.IMS_PER_BATCH 4    

Please note that:
1) If your model's name is different, please replace `Dynamic_ATSS_R_50_FPN_1x.pth` with your own.
2) If you enounter out-of-memory error, please try to reduce `TEST.IMS_PER_BATCH` to 1.
3) If you want to evaluate a different model, please change `--config-file` to its config file (in [configs/dynamic_atss](configs/dynamic_atss)) and `MODEL.WEIGHT` to its weights file.

## Training

The following command line will train Dynamic_ATSS_R_50_FPN_1x on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --config-file configs/dynamic_atss/dynamic_atss_R_50_FPN_1x.yaml \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/dynamic_atss_R_50_FPN_1x
        
Please note that:
1) If you want to use fewer GPUs, please change `--nproc_per_node` to the number of GPUs. No other settings need to be changed. The total batch size does not depends on `nproc_per_node`. If you want to change the total batch size, please change `SOLVER.IMS_PER_BATCH` in [configs/dynamic_atss/dynamic_atss_R_50_FPN_1x.yaml](configs/dynamic_atss/dynamic_atss_R_50_FPN_1x.yaml).
2) The models will be saved into `OUTPUT_DIR`.
3) If you want to train Dynamic ATSS with other backbones, please change `--config-file`.


## Citations
Please cite our paper in your publications if it helps your research:
```
@article{zhang2022dynamic,
  title={Dynamic Label Assignment for Object Detection by Combining Predicted and Anchor IoUs},
  author={Zhang, Tianxiao and Sharda, Ajay and Luo, Bo and Wang, Guanghui},
  journal={arXiv preprint arXiv:2201.09396},
  year={2022}
}
```
