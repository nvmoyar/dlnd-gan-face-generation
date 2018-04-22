# Deep Learning Foundations Nanodegree

[image1]: ./gan_screenshot.png "DCGAN Screenshot"


## Build a face generator with a GAN


### Project Overview

Use of Generative Adversarial Networks (GAN's) to generate new images of faces using MNIST and CelebA. Since the celebA dataset is complex, we want you to test the model on MNIST before CelebA. Running the GANs on MNIST will allow you to see how well the model trains sooner and it will make quick to tune our hyperparameters. 

The MNIST dataset contains images of handwritten digits in grey-scale. The CelebA is the abbreviation for CelebFaces Attributes Database dataset, which contains over 200,000 celebrity images with annotations. Since we will be generating faces, we do not need the annotations. Both datasets need to be preprocessed in order to feed our model. The values of the MNIST and CelebA dataset will be in the range of -0.5 to 0.5 of 28x28 dimensional images. The CelebA images will be cropped to remove parts of the image that do not include a face, then resized down to 28x28. This way the input tensor will have the same shape, unlike depth, since the MNIST images are black and white images with a single color channel, while the CelebA images have 3 color channels (RGB color channel). 

In order to get the Generative Adversarial Network model built, we need the following components, that will be implemented separately: 

* model_inputs: to create the TF Placeholders. Real input images are a tensor with rank 4. The learning rate will be a tensor with rank 0, therefore a scalar. We also define a TF Placeholder for the input noise vector z. This tensor will be used in the following way: the input images tensor will be fed to the discriminator, which is the component in charge to make the screen, and we feed the noise vector to the generator, the component in charge of doing the creative work.  

* generator and discriminator: we build them with the particularity of being able to reuse variables, and this is why a TensorFlow context manager is used. We want to reuse the parameters when we are generating the samples but that will not be during the training. Initially, the generator will start off by training and not by generating images. Both are a stack of convolutions - the discriminator is a convolution and the generator is a deconvolution-, and both use batch normalization. The generator uses the tanh function and this why during training, the vector noise is rescaled between [-1, 1]. 

* model_loss: we use label smoothing here since it has been proven as an enhancer of GAN's performance. One-sided label smoothing
Usually one would use the labels 0 (image is fake) and 1 (image is real). Using smoother labels (0.1 and 0.9) seems to make networks more resistant to adversarial since it helps to smooth the error calculation when the sigmoid function outputs a value near 0 or 1. [Further Reading](https://arxiv.org/abs/1606.03498). 

* model_opt: we need to bear in mind that we optimize both, generator and discriminator at the same time but [diferencing trainable parameters independently](https://www.tensorflow.org/programmers_guide/variables#sharing-variables). 

* train: put all together. It includes a component to print what is created by the generator every certain number of batches. That is useful in order to watch how fast the model is learning, thus how the generator is learning to fool the discriminator. During training we update both networks at the same time, so we need both losses. 

After tuning the model and prove it on the MNIST dataset, we proceed to generate faces feeding the discriminator with CelebA. The final predicted or generated faces are not so bad since most of the generated faces are actual human beings. However this model could be improved by using leaky relus instead of relus, that would increase the computation cost. Generally, leaky relu's are quite important because is the way that the generator can learn by receiving the gradient from the discriminator. Leaky relu outputs close to zero values for negative values instead of zero, which is an attempt to fix the dying relu problem. 

![DCGAN Screenshot][image1]

### Install environment, Project instructions and Test

* [Install instructions](https://github.com/udacity/deep-learning/face_generation/dlnd_face_generation.ipynb)
* [Test](http://localhost:8888/notebooks/dlnd-gan-face-generation/dlnd_face_generation.ipynb)
* [Demo](https://www.floydhub.com/nvmoyar/projects/face-generation)


#### Requirements

FloydHub is a platform for training and deploying deep learning models in the cloud. It removes the hassle of launching your own cloud instances and configuring the environment. For example, FloydHub will automatically set up an AWS instance with TensorFlow, the entire Python data science toolkit, and a GPU. Then you can run your scripts or Jupyter notebooks on the instance. 
For this project: 

> floyd run --mode jupyter --gpu --env tensorflow-1.0

You can see your instance on the list is running and has ID XXXXXXXXXXXXXXXXXXXXXX. So you can stop this instance with Floyd stop XXXXXXXXXXXXXXXXXXXXXX. Also, if you want more information about that instance, use Floyd info XXXXXXXXXXXXXXXXXXXXXX

#### Environments

FloydHub comes with a bunch of popular deep learning frameworks such as TensorFlow, Keras, Caffe, Torch, etc. You can specify which framework you want to use by setting the environment. Here's the list of environments FloydHub has available, with more to come!

#### Datasets 

With FloydHub, you are uploading data from your machine to their remote instance. It's a really bad idea to upload large datasets like CIFAR along with your scripts. Instead, you should always download the data on the FloydHub instance instead of uploading it from your machine.

> floyd run "python train.py" --data diSgciLH4WA7HpcHNasP9j

If you're using FloydHub, set data_dir to "/input" and use the FloydHub data ID "diSgciLH4WA7HpcHNasP9j".

Further Reading: [How and Why mount data to your job](https://docs.floydhub.com/guides/data/mounting_data/)

### Usage for this GAN project

floyd run --data udacity/datasets/celeba/1:data --data floydhub/datasets/mnist/1:data --mode jupyter

**You only need to mount the data that has been gently uploaded by Udacity and FloydHub, to your project's job.**

#### Output

Often you'll be writing data out, things like TensorFlow checkpoints, updated notebooks, trained models and HDF5 files. You will find all these files, you can get links to the data with:

> floyd output run_ID
