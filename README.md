# LAB1
Detect Pneumonia from chest X-Ray images

## Introduction
This lab is going to using Chest X-RAY images to classify Normal and Pneumonia classes by transfer learning method.We'll try to use some popular CNN pretrained models, such as ResNet50, InceptionV3 and EfficientNet as feature extractors and add the new classifier layers, or just directly use the pretrained model entirely by tuning the ouput size to binary. Additionally, we'll try to optimize the hyper-parameters, such as optimizer(ADAM, SGD, RMSprop), numbers of epochs, initial learning rate, and type of learning rate scheduler...etc.

## Datasets information
source：https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
1. Training dataset： 5,216
2. Testing datasets：   624
3. classes：2 classes (NORMAL, Pneumonia)

| numbers   | Training datasets | Testing datasets |
| :--------------: | :-----------------: | :----------------: |
| NORMAL           |     1341   |     234  | 
| PNEUMONIA        |     3875   |     390  |
| Total            |     5216   |     624  |
<img src="https://github.com/hcygeorge/aoi_defect_detection/blob/master/aoi_example.png" alt="alt text" width="360" height="300">


## Image transform

using the transforming and augmenting api from Pytorch with the training data.
https://pytorch.org/vision/stable/transforms.html

- RandomEqualize: p=10
- RandomRotation: degrees=(-25,20)
- CenterCrop: size=64

## Pretrained Model

1. ResNet50
2. InceptionV3
3. EfficientNetB0
4. EfficientNetB7

## Results

| Pretrained Model   | Training Accuracy | Testing Accuracy | F1-Score  |
| -------------------- | ------------------: | ----------------: | -------------: |
| ResNet50           |       100% |      94.4% |      94.9% |     94.9% |
| InceptionV3        |     95.07% |      94.4% |      94.9% |     94.9% |
| EfficientNetB0     |       100% |      94.4% |      94.9% |     94.9% |
| EfficientNetB7     |       100% |      94.4% |      94.9% |     94.9% |


## Reference

- Pytorch Models and pre-trained weight https://pytorch.org/vision/stable/models.html
- ResNet https://arxiv.org/abs/1512.03385
- InceptionV3 https://arxiv.org/abs/1512.00567
- EfficientNet https://arxiv.org/abs/1905.11946
