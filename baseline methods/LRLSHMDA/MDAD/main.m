for cv=1:10
    LRLSHMDA5cv(1,1,0.5)
    fold10(cv);
    overallauc(cv) = positiontooverallauc()
    overallaupr(cv) = pr_curve();
end
save overallauc overallauc
save overallaupr overallaupr
a = mean(overallaupr)
b = std(overallaupr)
c = mean(overallauc)
%save overallauc overallauc
%c = mean(overallauc)
%d = std(overallauc)