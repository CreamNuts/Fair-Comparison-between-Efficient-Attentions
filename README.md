# Fair Comparison between Efficient Attentions
This implementation is ans official code on the paper [*Fair Comparison between Efficient Attentions*](https://arxiv.org/abs/2206.00244) in [*CVPR 2022 Workshop on Attention and Transformers in Vision*](https://sites.google.com/view/t4v-cvpr22). In paper, we validated pyramid architecture with efficient attentions on ImageNet-1K. 

![poster](./poster.jpg)

## Requirements
```
conda env create -f environment.yml
```
Details are specified in `environment.yml`. Please be careful to install the pytorch. We did't test all the version of CUDA. 

## Usage
Our implementation depends on *[timm library](https://github.com/rwightman/pytorch-image-models)*. For usage, please refer to `train.py`.

* For single GPU training
    ```
    python3 train.py [data-dir] --model [model_name]
    ```
* For multi GPU training
    ```
    ./distributed_train.sh [number of gpu] [master port] [data_dir] --model [model_name]
    ```

## Public Reports
To learn more about the loss and learning process, click to the our [wandb project](https://wandb.ai/creamnuts/linear).
