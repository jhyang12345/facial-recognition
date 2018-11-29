# facial-recognition

## Data prep modules

### face_detector:
```bash
python face_detector.py -p <Directory-With-Images>
```

### faces_from_video:
```bash
python faces_from_video.py -p <Path-To-Video>
```

### prepare_dataset:
**Saves dataset in *datasets/positive* and *datasets/negative* to a numpy array file**  
(This is done to reduce the time spent by augmenting images in the dataset. The dataset is automatically loaded during train.py)
```bash
python -m data_prep.prepare_dataset
```

### augment_dataset:
**Augments image or directory of images according to the augmentations performed on the dataset used for this model**  
(The augmentations were focused on handling scale, blur, lighting, rotations, translations)
```bash
python -m data_prep.augment_dataset [-p <Path-To-Image> or -d <Directory-Of-Images>]
```


## Model training and run modules

### train.py
**Retrieve training and validation datasets and train specified model.**  
Models currently implemented for training are "hotdog", "cnn", "cnn_pool", "cnn_dropout"  
For more information view the models in the ***cnn_models*** directory
```bash
python train.py [-m <Model-Name> or --all]
```

### test.py
**Test trained model on image or directory of images.**
(Module mainly meant for jupyter notebooks a numpy array of squares drawn around detected face is plotted as a graph)
python test.py [-m <Model-Name> (-p <Path-To-Image> or -d <Directory-Of-Images>)]
