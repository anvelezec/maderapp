# Maderapp
The correct identification of timber species is a complicated task for the
wood industry and government institutions regulating the different laws
that ensure legal and transparent commerce. Currently, experts perform
this process using the organoleptic characteristics of the wood. However,
the methodology used is time-consuming and limited to environmen-
tal conditions. Moreover, it has a scalability issue since acquiring this 
specific knowledge and experience has a slow learning curve. On the
other hand, deep learning models have evolved as possible solutions for
process automation. Therefore, this paper explores convolutional neu-
ral network models suited to run on edge devices. The present study 
created a database with 25k images of 25 timber species from the
Peruvian Amazon. We trained-validated multiple lightweight models 
(less than 5M). The experiments were made using a repeated stratified
 k-fold cross-validation approach to estimate the performance of
the classifiers. The experiments show that the best model has an F1
score metric of 99.90% and 58ms latency using 561k parameters. Fur-
thermore, the created model showed an excellent ability to identify
species, opening up space for future integration with mobile appli-
cations, which helps minimize the time spent and the identification
errors on timber identification carried out by experts on control points.

## Data
This study used correctly identified wood core pieces and timber images from
principal commercial species in Peru to construct the dataset used to train
and validate the different convolutional neural network models. The timber images were
collected from Serfor trucks control transportation checkpoints located in La
Merced, Chanchamayo Province in the Jun ́ın Region in Peru. A cutting knife,
cell phone, and a portable microscope were used to capture multiple images
from different timber species. The portable microscope acquired images with a
magnification of 10x. No special treatment (e.g., pores filling or sanding) was
applied to wood samples.

The images to [train](https://storage.cloud.google.com/maderapp-images/training-img.zip?authuser=1) 
and [validate](https://storage.cloud.google.com/maderapp-images/validation.zip?authuser=1) and 
its [metadata](https://storage.cloud.google.com/maderapp-images/metadata.csv?authuser=5) can be dowloaded to replicate
the results.

## Neural Networks architectures
Neural networks (NN) applied to computer vision are mainly based on the 
concept of receptive fields, which enable the network to explore spatial correla-
tions in the image. As a result, neural networks are able to transform the crude
information in an input image into a more meaningful representation of its
content. However, nowadays, NN are growing larger and larger, which makes
them difficult to be used on edge devices since the existing hardware limita-
tions. Thus the use of a few parameter models is more suitable to work

under those devices conditions, for this NN has different particular components
which make them unique:

* Residual bottleneck: Wide-narrow-wide structure with the number of chan-
nels. The input has a high number of channels, which are compressed with 
a 1x1 convolution. The number of channels is then increased again with a
1x1 convolution so input and output can be added

* Inverted residual bottleneck: A narrow-wide-narrow approach, hence the
inversion. First, input is widened with a 1x1 convolution. Then a 3x3
depthwise convolution is used (which significantly reduces the number of
parameters). Finally, a 1x1 convolution is used to reduce the number of
channels so input and output can be added.


* Depthwise convolutions: Type convolution where a single convolutional filter
is applied for each input channel.

```python
in_channels = 3
depth_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, groups=in_channels)
```

* Pointwise convolutions: Type of convolution that uses a 1x1 kernel. This
kernel iterates through every single point and has a depth of however many
channels the input image has. It can be used in conjunction with depthwise 
convolutions to produce an efficient class of convolutions known as
depthwise-separable convolutions. 

```python
in_channels, out_channels = 3, 9
point_conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
```

In this work, we focus on the following models: MobileNetV2, EfficientNet-B0,
Residual NN with patches


## Training
Look a the folder train_executions for training examples with fold partirion and full images.

## Results

The results of the experiments are summarised in the table below. The F1-Scores were
scaled by 100 to contrast the values. 

| Model              | fold-1 | fold-2 | fold-3 | fold-4 | test  |
|--------------------|--------|--------|--------|--------|-------|
| Patches-ResNet     | 99.24  | 98.97  | 99.32  | 99.41  | 99.90 |
| MobileNetV2        | 99.29  | 99.49  | 99.43  | 99.21  | 99.50 |
| EfficientNet-B0    | 99.24  | 99.01  | 99.34  | 99.38  | 99.60 |
| EfficientNet-B0-NS | 99.21  | 99.08  | 99.05  | 98.97  | 92.60 |
| RestNet            | 98.91  | 98.81  | 98.94  | 98.63  | 99.00 |
