%   COV(X) or COV(X,Y) normalizes by (N-1) if N>1, where N is the number of
%   observations.  This makes COV(X) the best unbiased estimate of the
%   covariance matrix if the observations are from a normal distribution.
%   For N=1, COV normalizes by N.
%
%   COV(X,1) or COV(X,Y,1) normalizes by N and produces the second
%   moment matrix of the observations about their mean.  COV(X,Y,0) is
%   the same as COV(X,Y) and COV(X,0) is the same as COV(X).

% NOTE! N is the number of observations, i.e. rows

% Example 1 - Single Value Variables

x = [3 -1 1] 
exp_x = mean(x)

y = [2 4 1]
exp_y = mean(y)

xy = x.*y
exp_xy = mean(xy)

cov_matrix = cov(x,y,1)

% Example 2 - More Than Single Value Variables

X = [3,2; -1,-2; 1,2;] % Creates a 3x2 matrix of variable X

% The mean outputs a pair of values here instead of a scalar
% It does this by weighting the values of each column
% So it sums up the values in the first column and then divides by the
% number of rows, which in this case is 3
% Example, (3 + (-1) + 1)/3 = 1
% Then for second column, (2 + (-2) + 2)/3 = 2/3
% So exp_X = (1, 2/3)
exp_X = mean(X)

% If you want to calculate the mean by row then you add a 2 as the last
% parameter
exp_X_row = mean(X,2)

% If you want to calculate the overall weighted mean, i.e. scalar then you
% do
exp_X_weighted = mean(mean(X))

Y = [2,3; -4,-5; 2,3;]
exp_Y = mean(Y)

XY = X.*Y
exp_XY = mean(XY)

A = [X Y]

cov_matrixX = cov(X, 1)
cov_matrixY = cov(Y, 1)

% The cov matrix below will the cov and var of each variable of a variables
% so X = x1, x2 and Y = y1, y2
% To get the overall covariances between X and Y, 
% just add the cov of x1,y1 and x2,y2 from the outputed matrix
% We call the original cov matrix outputed by the syntax below the
% 'Pointwise Covariance Matrix'
cov_matrix2 = cov(A, 1)


