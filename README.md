# LAB1
Detect Pneumonia from chest X-Ray images

## Introduction

本專案目的為藉由AOI影像訓練深度學習模型辨識產品表面瑕疵，使用框架為Pytorch。實作結果顯示，預訓練VGG16模型的測試準確已達到99.0%。(目前排行榜上最高分為99.9%)
未來有時間會再嘗試更新的模型架構(如ResNet、DenseNet)，相信能進一步提升測試準確率。

## Datasets

本次影像資料是由工研院電光所在Aidea(人工智慧共創平台)釋出作為開放性議題，提供參賽者建立瑕疵辨識模型。但基於保密原則，平台並未透漏影像資料源自何種產線和產品。

資料來源：https://aidea-web.tw/topic/a49e3f76-69c9-4a4a-bcfc-c882840b3f27


## Datasets information

1. 訓練資料： 2,528張(隨機抽取20%作為驗證資料)
2. 測試資料：10,142張
3. classes：6 個類別(正常類別 + 5種瑕疵類別)



<img src="https://github.com/hcygeorge/aoi_defect_detection/blob/master/aoi_example.png" alt="alt text" width="360" height="300">


## Image transform

- 影像隨機水平翻轉(p=0.5)
- 影像隨機旋轉正負 15 度
- 影像大小縮放成 224 x 224

## Pretrained Model

1. ResNet50
2. InceptionV3
3. EfficientNetB0
4. EfficientNetB7

## 成果

下表為建模結果，可看出以預訓練VGG16輸入AOI影像訓練後的辨識結果最佳，對10,142張測試資料的準確度(Accuracy)已達到99%。測試資料的準確度是將預測結果上傳Aidea平台，由Aidea平台評分而得。  
| 模型結構           | 訓練準確率 | 驗證準確率 | 測試準確率 |
| ------------------ | ---------: | ---------: | ---------: |
| LeNet5             |      97.7% |      94.4% |      94.9% |
| VGG16              |     100.0% |      98.4% |      98.2% |
| VGG16 (pretrained) |     100.0% |      99.8% |      99.0% |


| Pretrained Model   | Training Accuracy | Testing Accuracy | F1-Score  |
| ------------------ | ---------: | ---------: | ---------: |---------: |
| ResNet50           |       100% |      94.4% |      94.9% |     94.9% |
| InceptionV3        |     95.07% |      94.4% |      94.9% |     94.9% |
| EfficientNetB0     |       100% |      94.4% |      94.9% |     94.9% |
| EfficientNetB7     |       100% |      94.4% |      94.9% |     94.9% |



## Reference

Pytorch Models and pre-trained weight  
https://pytorch.org/vision/stable/models.html   
