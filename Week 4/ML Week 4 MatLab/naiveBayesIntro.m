clear all;
% Below is the original dataset with the 5th column of each row being a 
% 1 (Scottish) and 0 (English).

X = [ 0 0 1 1 0 ;
1 0 1 0 0 ;
1 1 0 1 0 ;
1 1 0 0 0 ;
0 1 0 1 0 ;
0 0 1 0 0 ;
1 0 1 1 1 ;
1 1 0 1 1 ;
1 1 1 0 1 ;
1 1 1 0 1 ;
1 1 1 1 1 ;
1 0 1 0 1 ;
1 0 0 0 1 ];

Y = X(:,5); % Assign the class/label column (5th) to the Y vector
X = X(:,1:4)'; % X in proper format now, i.e. X just holds our features

% Below we have our priors, i.e. the class/label probabilities
pS = sum (Y)/size(Y,1);     % all rows with Y = 1, i.e. p(Scottish)
pE = sum(1 - Y)/size(Y,1);  % all rows with Y = 0, i.e. p(English)

% Below we have the given possibilities (biases) of our parameters, i.e. the
% class/labels. These will be used in our likelihood function later.
% phiS is P(X|Y=Scottish) so if you look at X and Y, you will notice X has
% 13 columns for each observation of one of the four features of X, and Y
% will have 13 rows for the class/label for each observation of one of the
% four features. 
% So for example, if we have X = 0 and Y = 1, we get 0 which means, for
% this feature observation we did not get a Scottish label. But if we have
% X = 1 and Y = 1 we get 1 which means, for this feature observation, we
% got a Scottish label.
% So if you look at xy, you will see that we have the total number of
% observations that led a Scottish label for each feature:
xy = X * Y;

% We then divide xy by sum(Y), which is the evidence for all Scottish
% labels, thus giving us P(X|Y=1), i.e. a likelihood of getting such
% feature observations given that the label is Scottish:
phiS = X * Y / sum(Y);  % all instances for which attrib phi(i) and Y are both 1
                        % P (X/Y=1)

% Here we have the exact same concept as above, but for English labels,
% thus we will have 0 for Scottish and 1 for English instead through (1-Y)
phiE = X * (1-Y) / sum(1-Y) ;  % all instances for which attrib phi(i) = 1 and Y = 0
                               % P(X/Y=0)

% Below is just our experiment outcome dataset.                              
x=[1 0 1 0]';  % test point 
              
% Bernoulli distribution: f = (p^k)((1-p)^(1-k))
% Remember our x here is a vector of binary values, so that 
% phiS.^x.*(1-phiS).^(1-x) below will yield a vector with each element being 
% a probability. 
% Then prod can be used to multiply all the values in x, as follows: 
pxS = prod(phiS.^x.*(1-phiS).^(1-x));
pxE = prod(phiE.^x.*(1-phiE).^(1-x));

% Finally, we have our posteriors for our classifications. Notice how
% summed they add up to 1, normalised by the denominator which will be the
% same for both.
pxSF = (pxS * pS ) / (pxS * pS + pxE * pE) %P(Y=1|X)
pxEF = (pxE * pE ) / (pxS * pS + pxE * pE) %P(Y=0|X) 