# dlnd_face_generation
Use of Generative Adversarial Networks (GAN's) to generate new images of faces using MNIST and CelebA. Since the celebA dataset is complex and you're doing GANs in a project for the first time, we want you to test your neural network on MNIST before CelebA. Running the GANs on MNIST will allow you to see how well your model trains sooner.

## Requirements

FloydHub is a platform for training and deploying deep learning models in the cloud. It removes the hassle of launching your own cloud instances and configuring the environment. For example, FloydHub will automatically set up an AWS instance with TensorFlow, the entire Python data science toolkit, and a GPU. Then you can run your scripts or Jupyter notebooks on the instance. 
For this project: 

> floyd run --mode jupyter --gpu --env tensorflow-1.0

You can see your instance in the list is running and has ID XXXXXXXXXXXXXXXXXXXXXX. So you can stop this instance with floyd stop XXXXXXXXXXXXXXXXXXXXXX. Also, if you want more information about that instance, use floyd info XXXXXXXXXXXXXXXXXXXXXX

### Environments

FloydHub comes with a bunch of popular deep learning frameworks such as TensorFlow, Keras, Caffe, Torch, etc. You can specify which framework you want to use by setting the environment. Here's the list of environments FloydHub has available, with more to come!

### Datasets 

With FloydHub, you are uploading data from your machine to their remote instance. It's a really bad idea to upload large datasets like CIFAR along with your scripts. Instead, you should always download the data on the FloydHub instance instead of uploading it from your machine.

> floyd run "python train.py" --data diSgciLH4WA7HpcHNasP9j

If you're using FloydHub, set data_dir to "/input" and use the FloydHub data ID "diSgciLH4WA7HpcHNasP9j".

For this project, diSgciLH4WA7HpcHNasP9j is the ID for this dataset.

### Output
Often you'll be writing data out, things like TensorFlow checkpoints. Or, updated notebooks. To get these files, you can get links to the data with:

> floyd output run_ID
