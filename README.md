# Build MobileNetV2-SSD from scratch

By combining MobileNetV2 with the Single Shot Detector, one can train an object detector super fast with high accuracy.

These scripts only shows how to build MobileNetV2-SSD layer by layer. The datasets and most of the parameters are hard-coded.
Doesn't have the functionality to resume training. 

The original SSD used VGG16 backbone. For our use, we will use MobileNetV2 as backbone.

## Understanding directory

`train.py`, `eval.py`, `utils.py` are for training, evaluation and helper functions.

`model.py` stores the definition of neural network architecture.

`coco.names` stores the index and label correspondence for object detection.

`001` directory contains training data. Every .jpg file should have a same name .txt file,
where .txt file is for storing the ground truth bounding box with corresponding labeling
as described in `coco.names`.

`002` directory contains evaluation data with groud truth. After evaluation, `eval.py` 
will draw ground truth boxes as green, and predicted boxes as yellow, then saved in `output` 
directory. User can obtain the result of evaluation visually.

`output` contains images after evaluation.

`models` directory contains a .pth model and .onnx model. Only .pth can be imported for 
resume training purpose. The .onnx model is for model architecture visualization purpose. 
To visualize the neural network, download the .onnx file and drag it to [netron](https://netron.app/).


## Run the script

``` bash
    # Notation: <Required> [Optional]

    # Training
    # Syntax
    python3 train.py -e [number of epochs]

    # Train for 200 epochs
    python3 train.py -e 200

    # Evaluation
    # Syntax 
    python3 eval.py -m <model path> -n [number of images]

    # Evaluate 200 images
    python3 eval.py -m models/'model_mobilenetv2_ssd_Epoch30_2021-03-31 15:22:28.531602.pth' -n 200
```


## About training

During training, the model will be saved every 10 epochs. These models can be found in `models/`.

## About evaluation

A timer has been implemented during evaluation. After finishing evaluation, output images with ground truth & prediction boxes, labels and confidences will be stored in `output/`.

## Inspect model structure

Find the file `models/'mobilenetv2_ssd_2021-04-01 11:37:31.317006.onnx'`, put it in [netron](https://netron.app/).


## Attention

This is a project I completed during my Internship in Recogine Technology, Malaysia. 
The dataset is coming from the company. 
Due to copyright issue, we can only provide some sample data for illustration purpose. 
If you're interested and want to use this repo, please refer to the dataset `001` and `002` and understand
how to prepare the data for training and validation. 
We gladly welcome any person to provide a rich free-to-use dataset with annotation. 
