clear
Trails = 100;
Heads = 0;
Tails = 0;
x = linspace(0,1,100)
figure; hold on; axis([0 1 0 10]);
prior = 1 + 0*x;
plotprior = plot(x,prior,'b');
leg = legend('prior');
for i = 1:Trails
    coin = rand();
    if coin >=0.5,
        Heads = Heads + 1;
        Like = Heads*x;
    else
        Tails = Tails + 1;
        Like = Tails - (x * Tails);
    end
    plotlike = plot(x,Like,'y');
    leg = legend('prior', 'likelihood');
    posterior = prior .* Like;
    trappost = trapz(posterior)/100;
    posterior = posterior / trappost;
    plotpost=plot(x, posterior, 'r');
    leg = legend('prior', 'likelihood', 'posterior');
    if(i < Trails),
        pause(1);
        delete(get(leg, 'Children'));
        delete(plotprior);
        delete(plotlike);
        delete(plotpost);
        prior = posterior;
        plotprior = plot(x,prior,'k');
        leg = legend('prior');
    end;
end