# Mnist Fashion Classifier
## Table of Contents : 
1. Used Dependencies 
2. Steps Done to prepreprocess the data 
3. Models I used for fitting the data
4. The chosen model 
5. Reciptive field of the model
6. MACCs and FLOPs per each layer of the model

## Used dependencies 
- Keras
- Numpy
- Matplotlib

## Models I used for fitting the data
#### At fisrt let's talk about the image generator I used :
1. The first one has these modifications on the data : rotation_range=8, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2 , actually I trained the model and used this image generator but it gave me less accuracy than training it without data augmentation but It's okey as it makes the model capable of detecting more patterns.
2. The second one has these modifications on the data : rotation_range=8, width_shift_range=0.02, height_shift_range=0.02, zoom_range=0.2  , but I think I does not make touchable modifications on the data , so I decided to exclude it.
#### Let's talk about the methodology of duilding the model
As I use Fashon MNIST dataset which consists of 60K sampel for training , I decided to train my own model from scratch at first and If Iam not satisfied with its performance I will go to fine-tuning.
The models I tried :
1. 1st model : consist of 2 CNNs of sizes 32 and 64 with 2 maxpooling layers, 2 dense layers of sizes 128 then 10 
2. 2nd model : same as the previous but I added another CNN of size 64 , and the number of max pooling layer is 2
3. 3rd model : I added another CNN of size 128 , and the number of max pooling layer is still 2
4. 4th model : Consists of 4 CNNs of sizes : 32 , 64 , 32, 64

Some techinques I use to improve the model performance :
1. Batch normalization
2. Dropout

Hyperparameters I tuned :
1. Number of epochs
2. Number of CNN layers and their number of filters
3. Number of Dense layers and their number of neurons

## The chosen model
The 2nd and the 3rd models roughly have the same results , but I decided to use the 3rd as it has better reciptive field.
model_2
![model_2](https://github.com/AhmedElgamiel/MnistClassifier/blob/main/hist2.png)
model_3
![model_3](https://github.com/AhmedElgamiel/MnistClassifier/blob/main/hist3.png)

## Reciptive field of the model
They reciptive field of the choosen model is :
layer output sizes given image = 28x28
Layer Name = conv1,  RF size =   3
Layer Name = conv2,  RF size =   5
Layer Name = pool1,  RF size =   6
Layer Name = conv3,  RF size =  10
Layer Name = conv4,  RF size =  14
Layer Name = pool2,  RF size =  16

Also in the notebook, I provided a function to calculate it
But how to increase the reciptive field ?
1. By adding more convolutional layers, as the model become deeper , the reciptive field increases as we can see from the formula , the RF accumulate as wee go deeper in the Network
2. By add pooling layers or higher stride convolutions : pooling layers acts as they suummarize a part of the image, so their reciptive field increase

## MACCs and FLOPs per each layer of the model
MACCs of model_3 :

MACC_conv1 = 3 * 3 * 1 * 28 * 28 *32 = 225792
MACC_conv2 = 3 * 3 *32 * 28 * 28 * 64 = 14.45 * 10^6
MACC_conv3 = 3 * 3 * 64 * 14 * 14 * 64 = 7.225 * 10^6
MACC_conv4 = 3 * 3 * 64 * 14 * 14 * 128 = 14.45 * 10^6
MACC_dense1 = (6272+1) * 128 = 802944
MACC_dense2 = (128+1) * 10 = 1290

FLOPs of model_3 :

FLOPs_conv1 = [(3 * 3 * 1 ) * 32 + 32 ] * [28 * 28] = 250880
FLOPs_conv2 = [(3 * 3 * 32) * 64 + 64] * [28 * 28] = 14.5 * 10^6
FLOPs_conv3 = [(3 * 3 * 64) * 64 + 64] * [14 * 14] = 7.42 * 10^6
FLOPs_conv4 = [(3 * 3 * 64) * 128 + 128] * [14 * 14] = 14.476 * 10^6
FLOPs_dense1 = (6272+1) * 128 = 802944
FLOPs_dense2 = (128+1) * 10 = 1290

We can see that conv_2 , conv_4 are the most computationally expensive layers.

We can reduce MACCs and FLOPs by reducing the number of parameters , this can be achieved by reducing the model complexity but this is not available for all times , so there are some other solutions such as : 

1. pooling , a (2*2) pooling layer reduces the size of the feature map by 1/2
2. Model Pruning :  compression technique where redundant network parameters are removed while trying to preserve the original accuracy (or other metric) of the network.







