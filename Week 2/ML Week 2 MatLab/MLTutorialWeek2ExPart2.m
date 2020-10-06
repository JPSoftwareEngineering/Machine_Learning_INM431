load fisheriris.mat

% The data set consists of 50 samples from each of three species of Iris flowers:
% Iris setosa, Iris virginica and Iris versicolor 
% Four features were measured from each sample: 
% the length and the width of the sepals and petals, in centimeters. 
% Based on the combination of these four features, Fisher developed a linear discriminant model 
% to distinguish the species from each other.

% View a frequency table
tabulate(species)

figure; % This command states that the figure below is a new one. Use this to view all figures.
gplotmatrix(meas, meas, species)

figure;
gscatter(meas(:, 3), meas(:, 4), species, 'rgb', 'osd')
xlabel('Petal Length')
ylabel('Petal Width')

[m, n] = size(meas) % Size() returns the rows and columns of the vector/matrix
[r, c] = size(species)

% Training and test percentage split
P = 0.70

rng('default')

% We use randperm() to get a row vector containing a random permutation 
% of the integers from 1 to the input, in our case m = 150 
% without repeating elements.
% Permutation just means rearrangement of a set.
shuffled_idx = randperm(m) 

% round() rounds the number to its closest integer,
% if 0.5 then it rounds up.
% Note! Because shuffled_idx is a vector, we can just refer to the rows,
% even if it is a column vector.
train_x = meas(shuffled_idx(1:round(P * m)), :); 
train_y = species(shuffled_idx(1:round(P * r)), :);

% For our test split, we add +1, because we want to start from 1 after
% our training split.
% And 'end' is just to the end of our row indexes.
test_x = meas(shuffled_idx(round(P * m) +1:end),:);
test_y = species(shuffled_idx(round(P * r) +1:end),:);

ClassTree = fitctree(train_x, train_y)

view(ClassTree)

view(ClassTree, 'mode', 'graph')

% predict() predicts a new set of labels/classification using the
% test_x data and the trained ClassTree model.
labels = predict(ClassTree, test_x)

% resuberror gives you the difference between the
% trained response and predicted response
resuberror = resubLoss(ClassTree)

% To improve the measure of the accuracy of our tree
% we perform cross-validation
% Cross-validation, by default, splits the training data into
% 10 random parts and trains 10 new trees, each one on nine parts
% of the data.
% It then examines the accuracy of each new tree on the data not
% included in training that tree.
% This is a better test for accuracy as it compares many trees.
cvrtree = crossval(ClassTree)

% The cvrtree gives a loss estimate from the cross-validation
% we use that loss estimate 
cvloss = kfoldLoss(cvrtree)