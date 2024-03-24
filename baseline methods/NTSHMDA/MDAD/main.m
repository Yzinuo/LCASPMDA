for cv=1:1
    NTSHMDA10cv(1,1,0.8,0.9,0.3,0.8,0.2)
    fold5(cv);
    overallauc(cv)  = positiontooverallauc()
    overallaupr(cv) = pr_curve();
end
save overallauc overallauc
save overallaupr overallaupr
a = mean(overallaupr)
c = mean(overallauc)

