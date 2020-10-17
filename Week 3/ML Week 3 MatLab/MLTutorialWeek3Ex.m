clear all; clc; close all;
rng('default');

% Although a coin will realistically have a fair probability of landing
% either heads or tails of 0.5, in the statistical world we assume that
% each coin has a slight bias. So for example, if a coin factory produces a
% 100 coins, then each coin will have a slight bias for heads or tails,
% i.e. a higher/lower probability of landing on heads/tails.
% Using linspace, we assume that there are only 101 discrete biases, i.e.
% probability values between 0 and 1:

coin_tosses = 100; % Number of coin tosses, i.e. experiments

bias_heads = 0.5; % Our assumed bias for getting heads, we just use 0.5 for simplicity

possible_coin_toss_biases = linspace(0,1,100); % Continuous possibilities of our bias

figure; axis([0 1 0 1]);

% We assume to have no reason to believe that a randomly picked coin is
% more likely to have any of the 101 values compared to the rest. 
% Therefore, we start our estimation of the bias by assigning it a 
% uniform prior distribution.
% The ones() method gives us a vector of 1s of the size of our
% possible_coin_toss_biases vector, and then we divide that by the size of
% possible_coin_toss_biases again to get even probabilities for each:

prior_uniform_coin_tosses = ones(length(possible_coin_toss_biases),1)/length(possible_coin_toss_biases);

% Generate a vector of random coin flips (1 = Heads; 0 = Tails):
% coin_flip_samples = randi([0, 1], [1, coin_tosses]); for when we do not have a
% bias

% The code below converts each coin flip to a 0 (Tails) if less than 0.5
% and to a 1 (Heads) if above 0.5
coin_flip_samples = double(rand(coin_tosses, 1) < bias_heads); % For when we have a bias

for k=1:coin_tosses
    % Calculate prior, likelihood, and evidence
    prior = prior_uniform_coin_tosses;
    % Remember! Although having conjugate priors is the best case scenario
    % for Bayes modelling, most experiments do not have conjugate priors.
    % Like in this case where the likelihood, i.e. distribution for the
    % data given our parameter estimates, will have a Binomial distribution
    % since our data will take only one of two categorical values, 0 or 1.
    % Thus the likelihood function below is for a Binomial distribution.
    likelihood = possible_coin_toss_biases'.^coin_flip_samples(k).*(1-possible_coin_toss_biases').^(1-coin_flip_samples(k));
    evidence = sum(likelihood .* prior);
    
    % Calculate the posterior distribution
    posterior = likelihood .* prior ./ evidence;
    
    % Dynamically plot the posterior distribution
    figure(1);
    plot(possible_coin_toss_biases', prior_uniform_coin_tosses);
    title(sprintf('Flip %d out of %d', k, coin_tosses));
    xlabel('Heads Bias'); ylabel('P(Heads Bias | Flips)');
    ylim([0 1]);
    set(gca,'Xtick',0:0.1:1);
    drawnow;

    % Make the posterior distribution the next prior distribution
    prior_uniform_coin_tosses = posterior;
end
