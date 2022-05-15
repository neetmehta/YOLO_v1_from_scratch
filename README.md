## YOLO_v1_from_scratch
From scratch implementation of YOLO v1

In this project YOLO v1 has two decoders as of now:
- Resnet101
- Darknet

To use resnet101, change backbone as 'resnet101' and change dimensions of S to (12,39) in kitti.yaml
To use darknet, change backbone as 'darknet' and change dimensions of S to (6,20) in kitti.yaml

## Data root

└───root
    ├───testing  
    │   └───image_2  
    └───training  
        ├───image_2  
        └───label_2  
