function fold5(num)
load interaction
A=interaction;
[nr,nm] = size(interaction);
load known
[pp1,qq1]=size(known); %pp1序列号  qq1对应的微生物序列号
load unknown
[pp2,qq2]=size(unknown);
x1=randperm(pp1)';   %打乱顺序
x2=randperm(pp2)';
F1=zeros(nr,nm);    %构建0矩阵
for cv=1:10
interaction1=A;
    if cv<10
        B1=known(x1((cv-1)*floor(pp1/10)+1:floor(pp1/10)*cv),:);
        B2=unknown(x2((cv-1)*floor(pp2/10)+1:floor(pp2/10)*cv),:);
        for i=1:floor(pp1/10)
            interaction1(B1(i,1),B1(i,2))=0;
        end
    else
        B1=known(x1((cv-1)*floor(pp1/10)+1:pp1),:);
        B2=unknown(x2((cv-1)*floor(pp2/10)+1:pp2),:);
        for i=1:pp1-floor(pp1/10)*9
            interaction1(B1(i,1),B1(i,2))=0;
        end
    end
    for i=1:nr
        sd(i)=norm(interaction1(i,:))^2;  %计算矩阵2范数
    end
    gamad=nr/sum(sd')*1;
    for i=1:nm
        sl(i)=norm(interaction1(:,i))^2;
    end
    gamal=nm/sum(sl')*1;
    for i=1:nr
        for j=1:nr
            pkd(i,j)=exp(-gamad*(norm(interaction1(i,:)-interaction1(j,:)))^2);
        end
    end
    for i=1:nm
        for j=1:nm
            km(i,j)=exp(-gamal*(norm(interaction1(:,i)-interaction1(:,j)))^2);
        end
    end 
    %计算高斯核相似性
    for i=1:nr
        for j=1:nr
            kd(i,j)=1/(1+exp(-15*pkd(i,j)+log(9999)));
        end
    end
    F=0.01*interaction1'+(0.01^2)*(km*interaction1'+interaction1'*kd);
    F=F';
    [b1,bb1]=size(B1);
    [b2,bb2]=size(B2);
    for i=1:b1
        F1(B1(i,1),B1(i,2))=F(B1(i,1),B1(i,2));
    end
    for i=1:b2
        F1(B2(i,1),B2(i,2))=F(B2(i,1),B2(i,2));    
    end
end
str=strcat('./predict result/Predict_result',num2str(num));
save(str,'F1')