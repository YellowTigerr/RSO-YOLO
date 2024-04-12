# RSO-YOLO
Improved YOLOv9 for Oriented Object Detection in Optical Remote Sensing Images

![图片](./obb/图片30.png)



# Installation
Please refer to [requirements.txt](./requirements.txt) for installation and dataset preparation.

This repo is based on [yolov9](https://github.com/WongKinYiu/yolov9). 

## Acknowledgements
在yolov9项目基础上添加了obb分支。
I have used utility functions from other wonderful open-source projects.

# Train a obb model
```shell
cd obb
python train.py
```

1.3 Make sure your dataset structure same as:
```
└── datasets
    └── your data
          ├── train
              ├── iamges
                  |────1.jpg
                  |────...
                  └────10000.jpg
              ├── labels
                  |────1.txt
                  |────...
                  └────10000.txt

          ├── val
              ├── images
                  |────10001.jpg
                  |────...
                  └────11000.jpg
                    
              ├── labels
                    |────10001.txt
                    |────...
                    └────11000.txt
```

```
label 格式
<object-class> <x1> <y1> <x2> <y2 <x3> <y3> <x4> <y4>
```

# Train a hbb model
refer to [yolov9](https://github.com/WongKinYiu/yolov9)

## More detailed explanation
主要补充数据读取、数据增强、添加角度预测分支、旋转框标签匹配策略、旋转框相关损失函数、非极大值抑制、obb评估、绘图等内容.
相关实现的细节和原理后续补充.

## 有问题反馈
在使用中有任何问题，建议先按照检查环境依赖项，检查使用流程是否正确，善用搜索引擎和github中的issue搜索框，可以极大程度上节省你的时间。

* 代码问题提issues


