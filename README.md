# facial-recognition
This project is a CNN based deep learning model focused on recognizing one person from an image of their face. To be more specific the task is to train a model to recognize IU(아이유) who is a famous KPOP singer. The focus of this project, is to treat the task of facial-recognition as a regular object classification task, under the presumption that only images of faces will be fed into the network. **Although I am aware of models such as DeepFace, or FaceNet that are designed specifically to handle face recognition, the objective of this project is to reach similar results using realtively normal CNN based networks**  
I collected all the data to train our model(decided not to use the CelebA dataset), wrote all the pipelines to properly align and cut out faces in our dataset.  
**(Demo YouTube video: [demo video](https://youtu.be/xvfkDnFHwiU))**

## Data preparation methodology
Preparing data the right way was the key factor in achieving our final validation accuracy (97%~). The final procedure can be summarized by the following.  
1. Raw images scraping (from Google images)
2. Cropping faces from frames of videos (mainly from YouTube)
3. Filtering out images that have too low resolution
4. Labeling collected images as positive as negative
5. Augmenting the data  

The two steps that ended up taking most of the work and experimenting were step 2 and 5

#### Cropping and aligning faces from images & video frames
The reason I felt the need for face alignment when cropping the images was because all the implementations of algorithms like FaceNet and DeepFace used examples of faces that were neatly rotated and cropped as input. Also aside from that, I was planning to augment the dataset we collected by using scale, rotations and translations, and there was necessity for cleanly cut images that can be augmented. Even after the augmentations, we needed to capture all the features that a person has, and that is why we needed a properly centered and scaled image.  

**Original Image *(Source: Google Images)***  
<img src="https://github.com/jhyang12345/facial-recognition/blob/master/examples/test2.jpg" width="300" alt="Original Image"/>  

**Aligned Image**  
<img src="https://github.com/jhyang12345/facial-recognition/blob/master/examples/output2.jpg"
width="300" alt="Aligned Image"/>

**Labeled Image classified by final trained model** *Pink boxes indicate positive match*  
(Model is trained to recognize unaligned images through the help of image augmentation)  
<img src="https://github.com/jhyang12345/facial-recognition/blob/master/examples/labeled.jpg"
width="300" alt="Aligned Image"/>  
More examples are in the [jupyter notebook](https://github.com/jhyang12345/facial-recognition/blob/master/presentation.ipynb) and [demo video](https://youtu.be/xvfkDnFHwiU)  

The final dataset that I decided to go with consisted of around 7000~ images. There were 3000~ positive labeled images, and 4000~ negative labeled images. At first, I thought of using the CelebA dataset as the main source of negative examples. However, after skimming through the images and reading a couple of blog posts I ended up with the conclusion that the dataset was a bit biased towards including Westerners (there was a very low ratio of Asian faces). Since the narrowed down objective was to recognize IU, it was crucial to train the model to differentiate Asians. Deep Learning models are only as good as the data you feed them, and I felt the need to collect my own dataset.  

The majority of time spent on this project was writing code to pipeline directories of images, videos, to their cropped and aligned versions. This step required heavily on the algorithm to find facial landmarks and bounding boxes. For this task I used the [Face Recognition](https://github.com/ageitgey/face_recognition) Python library. After a few tests, I noticed that it performed a lot more accurately than the OpenCV Haar Cascades algorithm (after all they use different approches). I made sure to skip multiple frames when using videos as input, to make sure the images collected vary more.  

Data augmentation was crucial to achieving the final validation accuracy(97%~). Before augmenting the dataset, validation accuracy struggled to get past 87%\~89% while training accuracy shot up to 99%\~. The final augmentation method ended up being focused at applying artificial lighting, scale change, rotations, and blur. These were chosen under the premise that real life photos are likely to show alterations like these.

## Data prep modules

### face_detector:  
Handles the task of finding the face and if the facial landmarks are properly found, aligns the face using the face aligner algorithm.
```bash
python face_detector.py -p <Directory-With-Images>
```

### faces_from_video:  
Applies the face_detector function to individual frames in a video. Skip frame should be specified to prevent iteration of each individual frame.
```bash
python faces_from_video.py -p <Path-To-Video>
```

### augment_dataset:
**Augments image or directory of images according to the augmentations performed on the dataset used for this model**  
(The augmentations were focused on handling scale, blur, lighting, rotations, translations)
```bash
python -m data_prep.augment_dataset [-p <Path-To-Image> or -d <Directory-Of-Images>]
```

### prepare_dataset:
**Saves dataset in *datasets/positive* and *datasets/negative* to a numpy array file**  
(This is done to reduce the time spent by augmenting images in the dataset. The dataset is automatically loaded during train.py)
```bash
python -m data_prep.prepare_dataset
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

## Model types  

### cnn_models/hotdog.py
**DeepDog Separable Convolution layer based CNN model**  
Based hugely on the [DeepDog](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3) model that was trained to recognize hotdogs. The original intention was to make a model slim enough to port onto an Android phone. This model takes use of Separable Convolutions to minimize the effects of lacking parameters in slim models. **This model uses ELU activations**  
### cnn_models/cnn.py
**Relatively Traditional CNN model**  
This is the relatively traditional cnn model. **This model uses ELU activations**    
### cnn_models/cnn_pool.py  
**CNN model with Max Pooling layers in between**  
**This model uses RELU activations**  

### cnn_models/ensemble.py  
**Ensemble model of a selected number or all the above trained models**  
A simple ensemble model of specified models, that are pretrained. **This model simply takes the average of the above model outputs.**
This model produces the highest accuracy so far.
