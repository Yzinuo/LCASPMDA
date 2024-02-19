for cv=1:10
    NTSHMDA10cv(1,1,0.7,0.9,0.3,0.8,0.2)
    fold5(cv);
    overallauc = positiontooverallauc()
    overallaupr = pr_curve();
% KATZHMDA5cv(1,1,0.01,2);
% [a,b,c,precision,recall]=positiontooverallauc();
% overallauc(cv)=a;
% lp=strcat('./PR/precision',num2str(cv));
% lr=strcat('./PR/recall',num2str(cv));
% save(lp,'precision');
% save(lr,'recall');
% lf=strcat('/ROC/fpr',num2str(cv));
% lt=strcat('/ROC/tpr',num2str(cv));
% save(lf,'b');
% save(lt,'c')
end
save overallauc overallauc
save overallaupr overallaupr
a = mean(overallaupr)
b = std(overallaupr)
c = mean(overallauc)