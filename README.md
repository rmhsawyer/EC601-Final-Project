# BU EC601-Final-Project
Project Description: an Andorid and PC App that translates user's expression into emojis.

Our application can detect 6 basic human expressions: happy, sad, surprise, neutral, fear, angry.

Project traits:

1. PC real-time expression detection

2. multiple faces expression detection both on PC and Android

3. image expression detection on Android

4. multiple expressions emoji matching(combination emojis like happy-surprise, angry-surprse) or sigle(happy) expression matching

5. our model architecture:

gray-scale 48x48 image (cropped) >> 3 sequential conv.layers (32 feat. maps per layer) >> maxpool >> 3 sequential conv.layers (64 feat. maps per layer) >> maxpool >> 3 sequential conv.layers (128 feat. maps per layer) >>maxpool >> 2 sequential dense layers (20% dropout) >> softmax output

Authors:

Minghe Ren (sawyermh@bu.edu)

Simin Zhai (siminz@bu.edu)

Tianhen Hu (tianheng@bu.edu)

Xueying Pan (xueying@bu.edu)

# Directory Descriptions:
emojis : pngs of emojis (we're updating more)

model: training models and opencv models (Models with various combinations were trained and evaluated using GPU computing g2.2xlarge on AWS)

datasets: training datasets and codes we used


# File descriptions:
1. The haar-cascade_frontalface_default.xml in OpenCV contains pre-trained filters and uses Adaboost to quickly find and crop the face.

2. real-time.py allows you to run real-time face and expression detection and save the predection values into emotion.txt

3. live-plotting.py allows you to draw data saved in emotion.txt

4. emojis.py -- This is our core code which finishes the expression detection and emojis matching.

5. model.h5 and model.json -- our trained model and all kinds of model parameters like weights

6. my_model.pb -- for android implantation

7. Nariz.xml -- open source 25x15 Nose detector computed with 7000 positive samples

8. AdvancedAndroid_emojify folder -- Android application package

# How to run this project:

In this directory, try:

python real-time.py haar-cascade_frontalface_default.xml

python emojis.py haar-cascade_frontalface_default.xml


