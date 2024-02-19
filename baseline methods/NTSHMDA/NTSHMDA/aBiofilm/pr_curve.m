function [auprNTSH]=pr_curve()
load prNTSH.mat
load reNTSH.mat
load accNTSH.mat
load interaction
[n,m] = size(interaction);
plot(reNTSH,prNTSH,':','LineWidth',1.4);
p1=flip(prNTSH);
r1=flip(reNTSH);
area1(1,1)=p1(1,1)*r1(1,1)/2;
for k=2:235060
    area1(1,k)=[p1(1,k-1)+p1(1,k)]*[r1(1,k)-r1(1,k-1)]/2;
end
auprNTSH=sum(area1);
save auprNTSH auprNTSH;
xlabel('Recall ');
ylabel('Precision');
title('PR curve');
