<p align="center"><img width="400" src="https://i.imgur.com/NvA2LVI.jpg"></p>

## Domain Transfer Network (DTN)

<p align="center"><img src="https://media.giphy.com/media/WooO2XCyF19Cef8gdQ/giphy.gif"></p>

This is an implementation of [Unsupervised Cross-Domain Image Generation](https://arxiv.org/abs/1611.02200) on Keras.

Contributors: [Deepak Prabakar](https://www.linkedin.com/in/deepak-prabakar/), [Sravan Jayati](https://www.linkedin.com/in/sravan-jayati/)

### Packages Required

- keras
- tensorflow
- opencv-python
- scikit-image
- scipy
- matplotlib
- pickle
- pydot
- tqdm

### Structure of the Project

```
DTN
│   README.md
|   .gitignore
│
└──digits
│   │   dtn_digits.py                           --> script for training digits DTN
|   |   mnist_classifier.py                     --> script for training an mnist classifier
|   |   mnist_clf.h5                            --> trained mnist classifier
│   │   svhn_encoder.h5                         --> encoder for digits DTN 
|   |   test_dtn_digits.py                      --> testing to get accuracy for mnist generation
|   |   train_svhn.py                           --> script for training an SVHN classifier and saving the encoder
│   └----------
│
└--face
|   |   dtn_face.py                             --> script for training faces DTN
|   |   no_faces.npy                            --> list of images from the source dataset where faces weren't found
|   |   source_list.pkl                         --> list of images from the source dataset excluding images from no_faces.npy
|   |
│   └---facenet                                 --> See reference 3
|   |   |   facenet_keras.h5                    --> encoder for faces DTN
│   │   |   haarcascade_frontalface_alt2.xml    --> haar cascade based face detector
|   |   |   model.py                            --> face encoder architecture
|   |   |   preprocessing.py                    --> preprocessing required for the keras-facenet model
│   │   └----------
|   └----------   
└----------
```
### Datasets

#### Faces
- Source Domain:    [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 
- Target Domain:    Generated Bitmoji images (See Reference 2)
#### Digits
- Source Domain:    [SVHN](http://ufldl.stanford.edu/housenumbers/)
- Target Domain:    [MNIST](http://yann.lecun.com/exdb/mnist/)

### Results
#### Faces
We used Google's Facenet as an alternative to Facebook's Deepface model to implement the encoder in our network.
We were able to achieve only limited results from DTN for faces. These are some of the best results from all the training configurations we tried. You can notice that several features such as facial shape, lip and hair color and eyeglasses have been successfully transferred to the generated image in some of these cases.

The first column represents the model's ability to learn an identity transformation for Bitmoji images while the second column represents its domain transformation capabilities.

<p align="center"><img width="800" src="https://i.imgur.com/GPkMemX.jpg"></p>

#### Digits
On digits, we were able to replicate the results from the paper for the most part. We chose to work with grayscale images instead of RGB. You can see that the identity transformation and the domain transformation have been learnt well by the model.

To evaluate the efficacy of the model, the images generated from SVHN images were tested on an MNIST classifier which was trained to have a test accuracy of 99.3%. MNIST images generated from SVHN's train and test set passed through the MNIST classifier with accuracies of 86.3% and 85.9% respectively. While these numbers are slightly lesser than those presented in the paper, the generated images are visually very similar to actual MNIST images.

<p align="center"><img width="800" src="https://i.imgur.com/9VxFs2T.png"></p>

### References

1. [Unsupervised Cross-Domain Image Generation](https://arxiv.org/abs/1611.02200)
2. [CS229 Final Project: Unsupervised Cross-Domain Image Generation](https://github.com/davrempe/domain-transfer-net)
3. [keras-facenet](https://github.com/nyoki-mtl/keras-facenet) (face detector, alignment and encoder)
