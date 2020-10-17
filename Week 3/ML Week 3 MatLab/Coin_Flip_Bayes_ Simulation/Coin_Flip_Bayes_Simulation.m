bias_heads = 0.3; %The bias of the coin
n = 15000; %Number of coin flips

bias_range = (0:0.01:1)'; %The range of possible biases

%Start with a uniform prior distribution:
p_bias_heads = ones(length(bias_range),1)/length(bias_range);
    
%Generate a vector of random coin flips (1 = Heads; 0 = Tails):
flip_series = double(rand(n,1) < bias_heads);

for k=1:n
    %Calculate prior, likelihood, and evidence
    prior = p_bias_heads;
    likelihood = bias_range.^flip_series(k).*(1-bias_range).^(1-flip_series(k));
    evidence = sum(likelihood .* prior);
    
    %Calculate the posterior distribution
    posterior = likelihood .* prior ./ evidence;
    
    %Dynamically plot the posterior distribution
    figure(1)
    plot(bias_range, p_bias_heads)
    title(sprintf('Flip %d out of %d', k, n))
    xlabel('Heads Bias'); ylabel('P(Heads Bias | Flips)')
    ylim([0 1])
    set(gca,'Xtick',0:0.1:1)
    drawnow

    %Make the posterior distribution the next prior distribution
    p_bias_heads = posterior;
end