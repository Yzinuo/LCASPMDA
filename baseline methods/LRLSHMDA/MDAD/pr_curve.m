function [auprLRLS]=pr_curve()
load prLRLS.mat
load reLRLS.mat
load accLRLS.mat
load interaction
[n,m] = size(interaction);
plot(reLRLS,prLRLS,':','LineWidth',1.4);
p1=flip(prLRLS);
r1=flip(reLRLS);
area1(1,1)=p1(1,1)*r1(1,1)/2;
for k=2:235060
    area1(1,k)=[p1(1,k-1)+p1(1,k)]*[r1(1,k)-r1(1,k-1)]/2;
end
auprLRLS=sum(area1);
save auprLRLS auprLRLS;
xlabel('Recall ');
ylabel('Precision');
title('PR curve');
