clear all;

% 1 (Male), 0 (Female)
% Feature columns are in the following order
% Height, Weight, Footsize
gender_dataset = [1, 6, 180, 12;
    1, 5.92, 190, 11;
    1, 5.58, 170, 12;
    1, 5.92, 165, 10;
    0, 5, 100, 6;
    0, 5.5, 150, 8;
    0, 5.42, 130, 7;
    0, 5.75, 150, 9;
    ];

X = gender_dataset(:, 2:4)';
Y = gender_dataset(:, 1);

male = X(:, 1:4);
female = X(:, 5:8);

% Our priors are:
pMale = 0.5;
pFemale = 0.5;

testBiases = [6, 180, 12]; % Test biases for our features, Height, Weight, Footsize

% Now, because our features are all continuous variables, we assume that
% they all have a Gaussian distribution. 
% So we need to get the parameters for our Normal distribution:
covFeaturesMale = cov(male', 1);

varHeightMale = covFeaturesMale(1, 1);
varWeightMale = covFeaturesMale(2, 2);
varFootsizeMale = covFeaturesMale(3, 3);

meanHeightMale = mean(male(1, :));
meanWeightMale = mean(male(2, :));
meanFootsizeMale = mean(male(3, :));

likeHeightMale = (2*pi*varHeightMale)^(-0.5)*exp(-0.5*(testBiases(1)-meanHeightMale).^2/varHeightMale);
likeWeightMale = (2*pi*varWeightMale)^(-0.5)*exp(-0.5*(testBiases(2)-meanWeightMale).^2/varWeightMale);
likeFootsizeMale = (2*pi*varFootsizeMale)^(-0.5)*exp(-0.5*(testBiases(3)-meanFootsizeMale).^2/varFootsizeMale);

% Repeat the above for females.
covFeaturesFemale = cov(female', 1);

varHeightFemale = covFeaturesFemale(1, 1);
varWeightFemale = covFeaturesFemale(2, 2);
varFootsizeFemale = covFeaturesFemale(3, 3);

meanHeightFemale = mean(female(1, :));
meanWeightFemale = mean(female(2, :));
meanFootsizeFemale = mean(female(3, :));

likeHeightFemale = (2*pi*varHeightFemale)^(-0.5)*exp(-0.5*(testBiases(1)-meanHeightFemale).^2/varHeightFemale);
likeWeightFemale = (2*pi*varWeightFemale)^(-0.5)*exp(-0.5*(testBiases(2)-meanWeightFemale).^2/varWeightFemale);
likeFootsizeFemale = (2*pi*varFootsizeFemale)^(-0.5)*exp(-0.5*(testBiases(3)-meanFootsizeFemale).^2/varFootsizeFemale);

% Finally, we can calculate our posteriors!
evidence = ((likeHeightMale.*likeWeightMale.*likeFootsizeMale).*pMale) + ((likeHeightFemale.*likeWeightFemale.*likeFootsizeFemale).*pFemale) 
posteriorMale = (pMale.*(likeHeightMale.*likeWeightMale.*likeFootsizeMale))/evidence;
posteriorFemale = (pFemale.*(likeHeightFemale.*likeWeightFemale.*likeFootsizeFemale))/evidence;

% To see if our classifier worked, we need to see if the posteriorMale is
% greater than the posteriorFemale. Since we used male test data, we expect
% the classifier to figure out that the data is for males.
isMale = posteriorMale > posteriorFemale; % 1 is True and 0 is False