# Machine_Learning_INM431

## Contents

[Week 1 Notes](#Week-1-Notes)

[Week 2 Notes](#Week-2-Notes)

## Week 1 Notes

### Key MatLab Commands

If you type any built in MatLab method after 'edit' in the MatLab command terminal, you will get detailed information on how the method works and even the code itself.

For example, the command below will give you detailed info on the 'cov' (covariance) method.

```
edit cov
```

### What is a Covariance Matrix?

A covariance matrix is a square matrix giving the covariance between each pair of elements of a given random vector. In the matrix diagonal there are variances, i.e., the covariance of each element with itself.

It essentially used as the covariance but in vector/matrix form. Just think of it as a table that tells you the covariance between variables.

## Week 2 Notes

## What are Parameters of a Model?

A model **parameter** is a constant configuration/setting variable that is internal to the model and whose value can be estimated from data. They are required by the model when making predictions and the values define the accuracy of the model on your problem.

## What are Hyperparameters of a Model?

A model **hyperparameter** is a configuration/setting that is external to the model and whose value cannot be estimated from data.

They are often used in processes to help **estimate** model parameters.
They are often specified by the practitioner.
They can often be set using heuristics.
They are often tuned for a given predictive modeling problem.

We cannot know the best value for a model hyperparameter on a given problem. We may use rules of thumb, copy values used on other problems, or search for the best value by trial and error.

When a machine learning algorithm is tuned for a specific problem, such as when you are using a grid search or a random search, then you are tuning the hyperparameters of the model or order to discover the parameters of the model that result in the most skillful predictions.

**If you have to specify a model parameter manually then it is probably a model hyperparameter.**

### Decision Trees

DTs are non-probabilistic, non-parametric supervised learning methods used for classification and regression problems. They are also the basis of random forests.

DTs learn from data to approximate/predict with a set of predicate logic rules, i.e. **if-then-else** decision rules. The deeper the tree, the more complex the decision rules and fitter the model.

A DT breaks down a dataset into smaller and smaller subsets while at the same time an associated DT is incrementally developed.

The final result is a tree with **decision nodes** and **leaf nodes**. So the decision nodes are basically the nodes asking the questions, and the leaf nodes represent a classification or prediction or final decision.

The top most decision node is also called the **root node**.

Example of a DT:

![alt text](DecisionTreeExample.png "Decision Tree Example")

The leaf nodes are the bottom nodes and all the others are decision nodes (also root node).

#### Decision Tree Structure in Detail

A DT consists of:

1. Nodes containing the names of independent variables
2. Branches labelled with the possible values of independent variables
3. Leaf nodes representing the classes, that is collection of observations grouped according to the values of one independent variable and joined to nodes via branches.

##### MATLAB Decision Tree Steps for Week 2 Tutorial Exercise Part 2

1. Starting from already classified data (training), we try to define some rules that characterise the various classes. 
2. After testing the model with a test set, the resulting descriptions (classes) are generalised (inference or induction) and used to classify records whose membership class is unknown.

Remember, specifically, nodes are labelled with the attribute name, i.e. variable name, branches are labelled with the possible values of the attribute/variable, and leaf nodes are labelled with the different values of the target attribute/variable.

An object is then classified by following a path along the tree that leads from the root to a leaf. Each path represents the rules of classification or prediction. 

To summarise:

1. Start from the root
2. Select the instance attributes associated with the current node
3. Follow the branch associated with the value assigned to that attribute in the instance
4. If you reach a leaf, return the label associated with the leaf, otherwise, beginning with the current node, repeat from step 2.

**Tutorial Exercise Steps**

**Read MATLAB FOR MACHINE LEARNING pages 2889 to 3022**

We use the Fisher Iris data set for this exercise.

The data set consists of 50 samples from each of three species of Iris flowers:

Iris setosa
Iris virginica
Iris versicolor 

Four features were measured from each sample: 

Sepal Length
Sepal Width
Petal Length
Petal Width 

in cm.

Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.

So basically we want to classify the species based on the size of its sepal and petal.

The 'meas' dataset has data for the length and width of the sepal and petal respectively (150x4). 
The 'species' dataset has data for the classification, i.e. species identifications. 

**So the rows from meas matches the row in species.**

Knowing this, we can first look at the species dataset through a frequency table:

```
tabulate(species)
```

![alt text](IrisFlowerSpeciesFrequencyTable.png "Iris Flower Species Frequency Table")

We can see that there are equal samples of each species, i.e. the data is equally distributed.

Let us also view the features of the species using a scatter plot matrix. The scatter plot matrix shows all the scatter plots of the species features in matrix format:

```
gplotmatrix(meas, meas, species)
```

The species dataset is added as our 'group' parameter. How this works is that the gplotmatrix() method matches each observation, i.e. row in the variable vectors/matrices, in our case the columns in meas, to each value in our group dataset, i.e. species. So for example, the first 50 variable values will be matched with the setosa species and grouped. You can see this in the legend from the graph image below:

![alt text](IrisFlowerDatasetScatterPlotMatrix.png "Iris Flower Dataset Scatter Plot Matrix")

Looking at the scatter plot matrix, we can also immediately tell that the setosa (blue) is different to the other two species, where the other two species seem to share similar features.

Now let us focus our visual analysis to compare just the petal measurements between species:

```
gscatter(meas(:, 3), meas(:, 4), species, 'rgb', 'osd')
xlabel('Petal Length')
ylabel('Petal Width')
```

The 'rgb' parameter set the colours for each value in the group variable, and the 'osd' parameter sets the symbol/mark for each group with osd standing for circle, square and diamond.

![alt text](IrisSpeciesPetalScatterPlot.png "IrisSpeciesPetalScatterPlot")

From the figure, we can see that a classification is definitely possible.

**So, let us create a classification tree (decision tree)!**

First we need to split our dataset into training/test. For this example, I have just used a 70% training split:

```
P = 0.70
```

We use randperm() to get a row vector containing a random permutation of the integers from 1 to the input, in our case m = 150, without repeating elements. Permutation just means rearrangement of a set:

```
shuffled_idx = randperm(m) 
```

The round() method rounds the number to its closest integer, if 0.5 then it rounds up. 

Note! Because shuffled_idx is a vector, we can just refer to the rows, even if it is a column vector.

```
train_x = meas(shuffled_idx(1:round(P * m)), :); 
train_y = species(shuffled_idx(1:round(P * r)), :);
```

For our test split, we add +1, because we want to start from 1 after our training split. And 'end' is just to the end of our row indexes:

```
test_x = meas(shuffled_idx(round(P * m) +1:end),:);
test_y = species(shuffled_idx(round(P * r) +1:end),:);
```

The 'fitctree()' method returns a fitted **binary** decision tree based on the input and output variables respectively.

```
ClassTree = fitctree(train_x, train_y)
```

There are two ways to view our decision tree:

1. The 'view(ClassTree)' method returns a text description of the tree.

```
view(ClassTree)
```

![alt text](IrisFlowersClassificationTreeText.png "Iris Flowers Classification Tree Text")

2. The view(ClassTree, 'mode', 'graph') method returns a graphic description of the tree.

```
view(ClassTree, 'mode', 'graph')
```

![alt text](IrisFlowersClassificationTreeGraph.png "Iris Flowers Classification Tree Graph")

Looking at the graph above, we can see that in fact only two input variables have been used to create our decision tree. These are x3 and x4, which are the length and widths of the petals. So the sepals are unused.

**Let us test our decision tree on new data!**

The predict() method predicts a new set of labels/classification using the test_x data and the trained ClassTree model:

```
labels = predict(ClassTree, test_x)
```

The 'predict()' method returns a vector/matrix of predicted class labels for the predictor data based on the trained classification tree, ClassTree in our case. 

**So how do we actually test the performance?**

We can first test it out by calculating the **resubstitution error**. This is the difference between the response training data and the predictions the tree makes of the response based on the input training data. It provides an initial estimate of the performance of the model, and it works only in one direction, in the sense that a high value for the resubstitution error indicates that the predictions of the tree will not be good. However, a low resubstitution error does not guarantee good predictions either, so it tells us nothing about it.

```
resuberror = resubLoss(ClassTree)
```

A resuberror of 0.0200 suggests that our tree classifies nearly all of our data correctly. 

To improve the measure of the predictive accuracy of our tree, we perform **cross-validation** of the tree. By default, cross-validation splits the training data into 10 parts at random. It trains 10 new trees, each one on nine parts of the data. It then examines the predictive accuracy of each new tree on the data not included in training that tree. 

Note! It cross-validates using the training data used to train our tree! So it is based on 105 observations, not the original 150! This is why it is different to splitting to train/test in the first place!

```
cvrtree = crossval(ClassTree)

cvloss = kfoldLoss(cvrtree)
```

At first, we used the crossval() method which performs an average loss estimate using cross-validation. **A cross-validated classification model was returned.** A number of properties were then available in MATLAB's work space. 

Then, we calculated the classification loss for observations not used for training by using the kfoldLoss() method, i.e. we are getting the actual loss figure from the cross-validated classification model. The low calculated value confirms the quality of the model, i.e. a low loss or difference between the prediction and actual and prediction.

**Finally, let us test our predicted labels against our test_y set**

We can do this using a **confusion matrix** to compare the test_y against the resulting labels:

```
[c order] = confusionmat(test_y, labels)
```

**How do we summarise all this together?**

In building any ML model, there are certain steps commonly followed such as:

1. Data preprocessing
2. Data partitioning into train/test
3. Training and evaluating

Technically, training is performed on train set, model is tuned on a validation set and evaluated on a test set.

**Validation set** is different from **test set**, in the sense that the validation set actually can be regarded as a part of the training set, because it is used to build your model, neural networks or others. It is usually used for parameter selection and to avoid overfitting. If your model is non-linear (like NN) and is trained on a training set only, it is very likely to get 100% accuracy and overfit, thus get very poor performance on the test set. Thus a validation set, which is independant from the training set, is  used for parameter selection. In this process, people usually use **cross validation**. 

So basically, the validation set is used for tuning the parameters of a model whilst the test set is used for performance evaluation. 
