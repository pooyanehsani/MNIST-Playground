# Dimension Reduction and Metric Learning on MNIST
In this project we choose two linear transformation, namely **Principle Component Analysis (PCA)** amd **Neighbourhood Component Analysis (NCA)**. **PCA** is a *unsupervised* metric learning approach which use feature space to construct the transformation matrix. On the other hand, 
**NCA** as a *supervised* metric learning method use both the label and the feature to learn the metric transformation. One of the advantage of considering label is it makes dense cluster in the representation space and dense clusters are useful for metric based classification like **SVM** and **KNN**.  NCA and PCA are implemented in python from scratch in this project to map the data point to a lower dimension representaion space and SVM and KNN are used to classify the test set to compare the efficiency of the transformation.
## Motivation
k-Nearest Neighbour (KNN) algorithms are among the simplest method in the machine learning algorithm. The idea behind the algorithm is to memorize the entire training set and use the label of the closest neighbor for the new instances to predict their label
Support Vector Machine (SVM) idea is to find the proper hyperplane to seprate the data. Obvously, there might be a lot of hyperplane which seprate the data but which one should be picked? SVM chose the one which maximize the marginal distance between points and the hyperplane.
Two methods use the metric space and distance function to learn the classification task and because of this, proper representation of the data in a low-dimension metric space is a must for a classifier. Dimensionallity reduction is the process of mapping a high dimensional data into a new space whose dimensionality is much smaller. PCA is one of the classic method for dimension reduction it computes new variable called *principal components* which are obtainedas linear combination of the original variable.  Component in PCA are orthogonal and in other word they areindependent and also we want to have the largest possible variance which can be achieved by choosing the eigen vector whith largest eigen value. NCA goal is to learn a transformation **A** where the KNN algorithm works well on the projected space **AX** where **X** is the data set. To achieve this NCA simply start with an arbitrary matrix for tranformation and a differentiable cost function where maximized with respect to the gradient. In this **MNIST Playground** I used MNIST data set as a benchmark to compare the NCA and PCA by the results of KNN and SVM on the transformed data. 

## 2D, 3D visualization
NCA 2D            |  NCA 3D
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/pooyanehsani/MNIST-Playground/master/images/NCA_2d.png)  |  ![](https://raw.githubusercontent.com/pooyanehsani/MNIST-Playground/master/images/NCA_3D.png)


PCA 2D            |  PCA 3D
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/pooyanehsani/MNIST-Playground/master/images/PCA_2d_legend.png)  |  ![](https://raw.githubusercontent.com/pooyanehsani/MNIST-Playground/master/images/PCA_3d.png)

As the above mentioned figures shows NCA make dense cluster compare to PCA and it is useful for KNN to make a correct prediction by looking to the neighbour of a new instance.

## Classification results

For the classification both NCA and PCA transform data to a 50 dimensional space and Linear-SVM and KNN trained on both to compare the results.

KNN         
:-------------------------:
![](https://raw.githubusercontent.com/pooyanehsani/MNIST-Playground/master/images/KNN.png)

SVM
:-------------------------:
![](https://raw.githubusercontent.com/pooyanehsani/MNIST-Playground/master/images/SVM.png)
