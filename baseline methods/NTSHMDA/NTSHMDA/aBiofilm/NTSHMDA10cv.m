function NTSHMDA10cv(gamadd,gamall,gamma,phi,delte,beta1,beta2)
%NTSHMDA10cv(1,1,0.7,0.9,0.3,0.8,0.2)
A=textread('known.txt');
nd=max(A(:,1)); 
nm=max(A(:,2));
[pp,qq]=size(A);
for i=1:pp
        interaction(A(i,1),A(i,2))=1;
end
save interaction interaction;
%5cv
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
        for i=1:nd
            for j=1:nd
                kd(i,j)=1/(1+exp(-15*pkd(i,j)+log(9999)));
            end
        end
      
  %reobtain adjacency matrix
  Am=interaction*km;
  Ad=kd*interaction;
     
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  based on Am   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
      %establish heterogeneous network based on reobtained adjacent matrix Am
      HN1=ones(nd+nm);
      HN1(1:nd,1:nd)=kd;
      HN1(1:nd,(nd+1):(nd+nm))=Am;
      HN1((nd+1):(nd+nm),1:nd)=Am';
      HN1((nd+1):(nd+nm),(nd+1):(nd+nm))=km;
  
      %the transition probability
      Am=Am.*interaction;
      W_md1=row_norm(Am');        %the transition probability from microbe nodes to disease nodes
      W_md1=W_md1';
      W_dm1=row_norm(Am);         %the transition probability from disease nodes to microbe nodes
      W_dm1=W_dm1';
     
  
      W_dd=row_norm(kd);         %the transition probability from disease nodes to disease nodes
      W_mm=row_norm(km);         %the transition probalibitly from microbe nodes to microbe nodes
 
      %transition probability matrix
      W1=HN1;
      
      %calculate disease-disease transition probability matrix
      for i=1:nd                  
         if sum(interaction(i,:))==0
              W1(i,1:nd)=W_dd(i,:);
         else
              W1(i,1:nd)=(1-phi)*W_dd(i,:);
         end
      end
      
      W1(1:nd,(nd+1):(nd+nm))=phi*W_md1;
      W1((nd+1):(nd+nm),1:nd)=phi*W_dm1;
      
      %calculate microbe-microbe transition probability matrix
      for j=1:nm
         if sum(interaction(:,j))==0
              W1(nd+j,nd+1:nd+nm)=W_mm(j,:);
         else
              W1(nd+j,nd+1:nd+nm)=(1-phi)*W_mm(j,:);
         end
      end
      
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  based on Ad  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
     %establish heterogeneous network based on reobtained adjacent matrix Ad
      HN2=ones(nd+nm);
      HN2(1:nd,1:nd)=kd;
      HN2(1:nd,(nd+1):(nd+nm))=Ad;
      HN2((nd+1):(nd+nm),1:nd)=Ad';
      HN2((nd+1):(nd+nm),(nd+1):(nd+nm))=km;
  
      %the transition probability
      Ad=Ad.*interaction;
      W_md2=row_norm(Ad');        %the transition probability from microbe nodes to disease nodes
      W_md2=W_md2';
      W_dm2=row_norm(Ad);         %the transition probability from disease nodes to microbe nodes
      W_dm2=W_dm2';
    
  
      %W_dd=row_norm(DS);         %the transition probability from disease nodes to disease nodes
      %W_mm=row_norm(MS);         %the transition probalibitly from microbe nodes to microbe nodes
 
      %transition probability matrix
      W2=HN2;
      
      %calculate disease-disease transition probability matrix
      for i=1:nd                  
         if sum(interaction(i,:))==0
              W2(i,1:nd)=W_dd(i,:);
         else
              W2(i,1:nd)=(1-phi)*W_dd(i,:);
         end
      end
      
      W2(1:nd,(nd+1):(nd+nm))=phi*W_md2;
      W2((nd+1):(nd+nm),1:nd)=phi*W_dm2;
      
      %calculate microbe-microbe transition probability matrix
      for j=1:nm
         if sum(interaction(:,j))==0
              W2(nd+j,nd+1:nd+nm)=W_mm(j,:);
         else
              W2(nd+j,nd+1:nd+nm)=(1-phi)*W_mm(j,:);
         end
      end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
      
      %initial walk probability matrix
      M0=eye(nd+nm); %p0 is a matrix, representing the initial probability matrix
      M0(1:nd,1:nd)=(1-delte)*(1/nd).*M0(1:nd,1:nd);
      M0(nd+1:nd+nm,nd+1:nd+nm)=delte*(1/nm).*M0(nd+1:nd+nm,nd+1:nd+nm);
      Pt1=M0;
      Pt2=M0;
      
      %random walk on the heterogeneous network
      for i=1:10  
         P1 = (1-gamma) * W1 * Pt1 + gamma * M0;
         P2 = (1-gamma) * W2 * Pt2 + gamma * M0;
         Pt1=P1;
         Pt2=P2;
      end
     
      P=(beta1*P1+beta2*P2)/(beta1+beta2);
 prediction=P(1:nd,nd+1:nd+nm);
 F=prediction;
 
 [size1B,size2B]=size(B);
% obtain the score of tested  disease-microbe interaction
for i=1:size1B
finalscore(i,1)=F(B(i,1),B(i,2));
end
% make the score of seed  disease-microbe interactions as zero
% finalscore = prediction_score(A(cv,2),A(cv,1));
for i=1:nd
    for j=1:nm
       if interaction(i,j)==1
          F(i,j)=-10000;
        end
    end
end
for qq=1:size1B
[ll1,mm1]=size(find(F>=finalscore(qq)));
[ll2,mm2]=size(find(F>finalscore(qq)));
position(1,T)=ll2+1+(ll1-ll2-1)/2;
T=T+1;
end
end
save('position.mat','position'); 

 
end