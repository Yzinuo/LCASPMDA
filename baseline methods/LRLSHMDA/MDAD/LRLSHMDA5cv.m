function LRLSHMDA5cv(gamadd,gamall,lw)
%LRLSHMDA10cv(1,1,0.5)
A=textread('known.txt');
% nd:the number of diseases
% nm:the number of microbe
% pp:the number of known diseae-microbe associations
nd=max(A(:,1)); 
nm=max(A(:,2));
[pp,qq]=size(A);
%interaction: adjacency matrix for the disease-microbe association network
%interaction(i,j)=1 means microbe j is related to disease i
for i=1:pp
        interaction(A(i,1),A(i,2))=1;
end

save interaction interaction;

%implement 10-fold cross validation
x=randperm(pp)';
T=1;

for cv=1:10
load interaction interaction;

    if cv<10
        B=A(x((cv-1)*floor(pp/10)+1:floor(pp/10)*cv),:);
% obtain training sample
for i=1:floor(pp/10)
        interaction(B(i,1),B(i,2))=0;
    end
    else B=A(x((cv-1)*floor(pp/10)+1:pp),:);
        % obtain training sample
for i=1:pp-floor(pp/10)*9
        interaction(B(i,1),B(i,2))=0;
    end
    end
    
   
%calculate gamad for Gaussian kernel calculation
    for i=1:nd
        sd(i)=norm(interaction(i,:))^2;
    end
    gamad=nd/sum(sd')*gamadd;
    
 %calculate gamal for Gaussian kernel calculation
    for i=1:nm
        sl(i)=norm(interaction(:,i))^2;
    end
    gamal=nm/sum(sl')*gamall;
    
    %calculate Gaussian kernel for the similarity between disease: kd
    for i=1:nd
        for j=1:nd
    pkd(i,j)=exp(-gamad*(norm(interaction(i,:)-interaction(j,:)))^2);
        end
    end
    
    %calculate Gaussian kernel for the similarity between microbe: km
        for i=1:nm
            for j=1:nm
                km(i,j)=exp(-gamal*(norm(interaction(:,i)-interaction(:,j)))^2);
            end
        end 
% 
        for i=1:nd
            for j=1:nd
                kd(i,j)=1/(1+exp(-15*pkd(i,j)+log(9999)));
            end
        end
for i=1:nm
    for j=1:nm
   D(i,j)=0;
    end
end
DD=sum(km);
for i=1:nm
D(i,i)=DD(1,i);
end
km1=(D^(-1/2))*(D-km)*(D^(-1/2));


for i=1:nd
    for j=1:nd
   D1(i,j)=0;
    end
end
DD1=sum(kd);
for i=1:nd
D1(i,i)=DD1(1,i);
end
kd1=(D1^(-1/2))*(D1-kd)*(D1^(-1/2));

FM=km*pinv(km+km1*km)*interaction';
FD=kd*pinv(kd+kd1*kd)*interaction;
F=lw*FM'+(1-lw)*FD;
% F=F';
%      F=F';

[size1B,size2B]=size(B);
% obtain the score of tested  disease-microbe interaction
for i=1:size1B
finalscore(i,1)=F(B(i,1),B(i,2));
end
% make the score of seed  disease-microbe interactions as zero
for i=1:nd
    for j=1:nm
        if interaction(i,j)==1
           F(i,j)=-10000;
        end
    end
end


for qq=1:size1B
% obtain the position of tested disease-microbe interaction as variable position(1,cv), 
[ll1,mm1]=size(find(F>=finalscore(qq)));
[ll2,mm2]=size(find(F>finalscore(qq)));
position(1,T)=ll2+1+(ll1-ll2-1)/2;
T=T+1;
end

end
save('position.mat','position');  

end



        
        
        
    
   



