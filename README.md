# Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

##1. Overview
The goal of this project is to build the traffic sign recognition successfully.

The dataset is from [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, 5 test images picked from internet are used to test the model accuracy.

The goals / steps of this project are the following:

- Load the data set
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report



Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

##2. Project Preparation

1. Dependencies

   This lab requires [CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

   The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

2. Dataset and Repository

3. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.

4. Clone the project, which contains the Ipython notebook and the writeup template.

```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

##3. Project Introduction
This project is using the convolutional neural network to classify the German traffic sign. The advantage of using the CNN is that it can improve the accurary based on the huge dataset comparing the traditional image process, such as OpenCV.
The model of CNN in this project is based on LeNet-5 from the class.

##4. Project Pipeline
The workflow of this project is followed by the pipeline below:

1. *Design and Test a Model Architecture*
	1. Preprocessing
	2. Model Architecture
	3. Model Training
	4. Solution Approach
2. *Test a Model on New Images*
	1. Acquiring New Images
	2. Performance on New Images
	3. Model Certainty - Softmax Probabilities

###4.1 Design and Test a Model Architecture
####4.1.1 Preprocessing

1. Data Load

   The convolutional neural network needs quite a lot of data.  [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) provides the examples and the labels. The first block is to read the training examples, the validation examples and the test examples.

   ````python
   training_file = './train.p'
   validation_file= './valid.p'
   testing_file = './test.p'

   with open(training_file, mode='rb') as f:
       train = pickle.load(f)
   with open(validation_file, mode='rb') as f:
       valid = pickle.load(f)
   with open(testing_file, mode='rb') as f:
       test = pickle.load(f)
       
   X_train, y_train = train['features'], train['labels']
   X_valid, y_valid = valid['features'], valid['labels']
   X_test, y_test = test['features'], test['labels']
   ````

   The groups of different examples are ready. `X_train` and `y_train` are for training examples; `X_valid` and `y_valid` are for validation examples; and `X_test` and `y_test` are for test examples.

   The size of each group can also be extracted by simply using `numpy`:

   ````python
   n_train = len(X_train)
   n_validation = len(X_valid)
   n_test = len(X_test)
   image_shape = X_train[0].shape
   n_classes = len(np.unique(y_train))
   ````

   ````
   # Output
   Number of training examples = 34799
   Number of testing examples = 12630
   Image data shape = (32, 32, 3)
   Number of classes = 43
   ````

2. Data Visualization

   Visualizing the examples is easier for understanding and for preprocessing. The visualization is separated into 3 parts: 

   * Randomly plot 1 training example and its class
   * Plot the number of traffic sign in the training examples and in the test examples
   * Plot all the classes with classes' number and sign name

   ````python
   import random
   import matplotlib.pyplot as plt

   %matplotlib inline

   index = random.randint(0, len(X_train))
   image = X_train[index].squeeze()
   plt.figure(figsize=(1,1))
   plt.imshow(image)
   print(y_train[index])
   ````

   ````
   # Output
   11
   ````

   ![random_plot](D:\temp\Traffic-Sign-Classifier-Project-master\random_plot.png)

   The motivation to plot the number of traffic sign type is to check if there is uneven distribution. The uneven distribution will have impact on the accurary if the regularization is missiong. For instance, if the data set has much more example on the class 11 than on the class 1, the prediction will be more likely to predict the class 11.

   ![number_traffic_sign_type](D:\temp\Traffic-Sign-Classifier-Project-master\number_traffic_sign_type.png)

   The result shows that the distribution is uneven, *bias* is necessary.

   The following figure shows the classes.

   ````python
   rows, cols = 4, 12
   # ax_array is a array object consistint of plt object
   fig, ax_array = plt.subplots(rows, cols) 
   plt.suptitle('Training set classes')
   for class_idx, ax in enumerate(ax_array.ravel()):
       if class_idx < n_classes:
           cur_X = X_train[y_train == class_idx]
           cur_img = cur_X[np.random.randint(len(cur_X))]
           ax.imshow(cur_img)
           ax.set_title('{:02d}'.format(class_idx))
       else:
           ax.axis('off')
   # hide both x and y ticks
   plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
   plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)

   plt.draw()
   ````

   <figure is missing>

3. Pre-process the data set

   The following technique is used to pre-process the data set:

   * Normalization and RGB2GREY
   * More data generation

   2 methods of normalization are tried. First one is using `cv2.createCLAHE` 

   ````python
   ### Normalize data
   ### Convert input from range 0 ~ 255 to -128 ~ 127
   X_train_cl = X_train_cl.astype(int)
   X_test_cl = X_test_cl.astype(int)
   X_train_cl -= 128
   X_test_cl -= 128
   print(X_train_cl.dtype)
   print(np.unique(X_train_cl[2]))
   ````

   ​

####4.1.2 Model Architecture



####4.1.3 Model Training

####4.1.4 Solution Approach