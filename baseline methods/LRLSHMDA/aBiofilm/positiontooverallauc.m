function [overallauc,fpr,tpr,prLRLS,reLRLS]=positiontooverallauc()
load position.mat;
load interaction;
[n,m]=size(interaction);
sID=textread('known.txt');
[pp,qq]=size(sID);
k1=m*n;

for k=1:m*n-floor(pp/10)*9
    tp=0;
    for t=1:pp
        if position(1,t)<=k
            tp=tp+1;
        end
    end
    tpr(1,k)=tp/pp;
    if k<m*n-pp+floor(pp/10)+1
    fp=k*pp-tp;
    else fp=floor(pp/10)*9*(m*n-pp+floor(pp/10))+(pp-floor(pp/10)*9)*k-tp;
    end

     fpr(1,k)=fp/(floor(pp/10)*9*(m*n-pp+floor(pp/10)-1)+(pp-floor(pp/10)*9)*(m*n-floor(pp/10)*9-1));
end
for t1=1:k1
    [n1,m1]=size(position);
    tp=0;
    fp=0;
    tn=0;
    fn=0;
    for i=1:m1
        if position(1,i)<=t1
            tp=tp+1;
        end
    end
    fp=m1-tp;
    fn=t1-m1;
    tn=n*m-m1-fn;
    pr(t1)=tp/(tp+fp);
    re(t1)=tp/(tp+fn);
    acc(t1)=(tp+tn)/(tp+tn+fp+fn);
end
prLRLS=pr(:,2884:k1);
reLRLS=re(:,2884:k1);
accLRLS=acc(:,2884:k1);
save prLRLS prLRLS   %精确度
save reLRLS reLRLS   %召回率
save accLRLS accLRLS   %准确率
save tpr tpr
save fpr fpr
plot(fpr,tpr);

clear area;
area(1,1)=tpr(1,1)*fpr(1,1)/2;
for k=2:m*n-floor(pp/10)*9
    area(1,k)=[tpr(1,k-1)+tpr(1,k)]*[fpr(1,k)-fpr(1,k-1)]/2;
end
overallauc=sum(area);
end
          


