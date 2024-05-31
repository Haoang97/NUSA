# NUSA
This is a code repository for an IJCV paper "Does Confusion Really Hurt Novel Class Discovery?" by Haoang Chi, Wenjing Yang, Feng Liu, Long Lan, & Bo Han.
___
This paper introduces a new setting called Novel Class Discovery under Unreliable Sampling (NUSA), which addresses the problem of category confusion during the sampling process for novel class discovery (NCD). In NUSA, collectors may misidentify known classes and confuse them with novel classes, leading to unreliable NCD results. The authors propose a solution called the Hidden-Prototype-based Discovery Network (HPDN) to handle NUSA. HPDN aims to obtain clean data representations despite the confusedly sampled data and employs a mini-batch K-means variant for robust clustering, which alleviates the negative impact of residual errors in the representations by detaching the noisy supervision in a timely manner.

![setting_diagram](https://github.com/Haoang97/NUSA/blob/main/images/setting.png)

## Data preparation
For CIFAR-10 and CIFAR-100, you can download them [here](https://www.cs.toronto.edu/~kriz/cifar.html).

For ImageNet 2012, you can download it [here](https://www.image-net.org/).

For CUB, you can download it [here](https://www.vision.caltech.edu/datasets/cub_200_2011/).

For Stanford Cars, you can download it [here](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset).

## Hidden-prototype-based Discovery Network
First, the pretrained backbones can be downloaded from https://github.com/google-research/simclr?tab=readme-ov-file and converted to Pytorch format with https://github.com/tonylins/simclr-converter. Then, download these datasets and move them to a folder.

Step 1 & Step 2:
```
CUDA_VISIBLE_DEVICES=[gpu_ids] python main.py \
  --mode train \
  --dataset_name cifar10 \
  --dataset_root [your dataset dir] \
  --exp_root [your results saving dir] \
  --encoder_dir [your encoder dir] \
  --noise_rate 0.2 \
  --corss_rate 1.0 \
  --num_labeled_classes 5 \
  --num_unlabeled_classes 5
```
If you only want to do clustering,
```
CUDA_VISIBLE_DEVICES=[gpu_ids] python main.py \
  --mode test \
  --dataset_name cifar10 \
  --dataset_root [your dataset dir] \
  --exp_root [your results saving dir] \
  --encoder_dir [your encoder dir] \
  --noise_rate 0.2 \
  --corss_rate 1.0 \
  --num_labeled_classes 5 \
  --num_unlabeled_classes 5
```

## Cite this article
```
@article{chi2024does,
  title={Does Confusion Really Hurt Novel Class Discovery?},
  author={Chi, Haoang and Yang, Wenjing and Liu, Feng and Lan, Long and Han, Bo},
  journal={International Journal of Computer Vision},
  pages={1--17},
  year={2024},
  publisher={Springer}
}
```

## Acknowledgment
This work was supported by the National Natural Science Foundation of China (No. 91948303-1, No. 62372459, No. 62376282). We would like to thank the editor and reviewers for their valuable comments that were very useful for improving the quality of this work.

If you have any questions, feel free to contact haoangchi618@gmail.com.
