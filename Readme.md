## Domain Transfer Network (DTN)

This is an implementation of [Unsupervised Cross-Domain Image Generation](https://arxiv.org/abs/1611.02200) on Keras.

Contributors: [Deepak Prabakar](https://www.linkedin.com/in/deepak-prabakar/), [Sravan Jayati](https://www.linkedin.com/in/sravan-jayati/)

### Packages Required

- keras
- tensorflow
- opencv
- skimage
- matplotlib
- pickle

### Structure of the Project

```
DTN
│   README.md
|   .gitignore
│
└──digits
│   │   dtn_digits.py                           --> script for training digits DTN
│   │   svhn_encoder.h5                         --> encoder for digits DTN 
│   └----------
│
└--face
|   |   dtn_face.py                             --> script for training faces DTN
|   |   no_faces.npy                            --> list of images from the source dataset where faces weren't found
|   |   source_list.pkl                         --> list of images from the source dataset
|   |   target_list.pkl                         --> list of images from the target dataset
|   |
│   └---facenet
|   |   |   facenet_keras.h5                    --> encoder for faces DTN
│   │   |   haarcascade_frontalface_alt2.xml    --> haar cascade based face detector
|   |   |   model.py
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


### References

1. [Unsupervised Cross-Domain Image Generation](https://arxiv.org/abs/1611.02200)
2. [CS229 Final Project: Unsupervised Cross-Domain Image Generation](https://github.com/davrempe/domain-transfer-net)
3. [keras-facenet](https://github.com/nyoki-mtl/keras-facenet) (face detector, alignment and encoder)
