% function [overallauc,fpr,tpr]=positiontooverallauc()
% load position.mat;
% load interaction;
% [n,m]=size(interaction);
% sID=textread('known.txt');
% [pp,qq]=size(sID);
% k1=m*n;
% 
% 
% for k=1:m*n-floor(pp/10)*9
%     tp=0;
%     for t=1:pp
%         if position(1,t)<=k
%             tp=tp+1;
%         end
%     end
%     tpr(1,k)=tp/pp;
%     if k<m*n-pp+floor(pp/10)+1
%     fp=k*pp-tp;
%     else fp=floor(pp/10)*9*(m*n-pp+floor(pp/10))+(pp-floor(pp/10)*9)*k-tp;
%     end
% 
%      fpr(1,k)=fp/(floor(pp/10)*9*(m*n-pp+floor(pp/10)-1)+(pp-floor(pp/10)*9)*(m*n-floor(pp/10)*9-1));
% end
% for t1=1:k1
%     [n1,m1]=size(position);
%     tp=0;
%     fp=0;
%     tn=0;
%     fn=0;
%     for i=1:m1
%         if position(1,i)<=t1
%             tp=tp+1;
%         end
%     end
%     fp=m1-tp;
%     fn=t1-m1;
%     tn=n*m-m1-fn;
%     pr(t1)=tp/(tp+fp);
%     re(t1)=tp/(tp+fn);
%     acc(t1)=(tp+tn)/(tp+tn+fp+fn);
% end
% prLRLS=pr(:,2470:k1);
% reLRLS=re(:,2470:k1);
% accLRLS=acc(:,2470:k1);
% save prLRLS prLRLS   %精确度
% save reLRLS reLRLS   %召回率
% save accLRLS accLRLS   %准确率
% save tpr tpr
% save fpr fpr
% plot(fpr,tpr);
% clear area;
% area(1,1)=tpr(1,1)*fpr(1,1)/2;
% for k=2:m*n-floor(pp/10)*9
%     area(1,k)=[tpr(1,k-1)+tpr(1,k)]*[fpr(1,k)-fpr(1,k-1)]/2;
% end
% overallauc=sum(area);
% end
          

function [overallauc, fpr, tpr, accuracy, f1_score, mcc_score] = positiontooverallauc()
    load position.mat;
    load interaction; 
    load predict_result/Predict_result1.mat;  % 修改加载路径
    load val_pos1.mat
    load val_neg1.mat

    [n, m] = size(interaction);
    sID = textread('known.txt');
    [pp, qq] = size(sID);
    k1 = m * n;
    
    pos_indices = sID;
    
    predicted_scores_B1 = F1(sub2ind(size(F1), B1(:,1), B1(:,2)));
    predicted_scores_B2 = F1(sub2ind(size(F1), B2(:,1), B2(:,2)));

    predicted_scores = vertcat(predicted_scores_B1, predicted_scores_B2);
    disp(size(predicted_scores) );

    labels_B1 = interaction(sub2ind(size(interaction), B1(:,1), B1(:,2))); % 将true_labels保持为逻辑行向量
    labels_B2 = interaction(sub2ind(size(interaction), B2(:,1), B2(:,2)));
    labels = vertcat(labels_B1, labels_B2);
    disp(size(labels) );
    labels = labels.';
   
    predicted_scores = predicted_scores.';
    max_val = max(predicted_scores(:));
    min_val = min(predicted_scores(:));
    threshold = (max_val + min_val)/2;

    predicted_labels = (predicted_scores > threshold);
    disp(size(predicted_labels) );
    disp(size(labels) );
    TP = sum(predicted_labels == 1 & labels == 1);
    TN = sum(predicted_labels == 0 & labels == 0);
    FP = sum(predicted_labels == 1 & labels == 0);
    FN = sum(predicted_labels == 0 & labels == 1);
% %    
%     max_val = max(position(:));
%     min_val = min(position(:));
%     t1 =(max_val + min_val )/2;
%     [n1,m1]=size(position);
%     TP=0;
%     FP=0;
%     TN=0;
%     FN=0;
%     for i=1:m1
%         if position(1,i)<=t1
%             TP=TP+1;
%         end
%     end
%     FP=m1-TP;
%     FN=t1-m1;
%     TN=n*m-m1-FN;
    disp(TP)
    disp(TN)
    disp(FP)
    disp(FN)
    % 计算Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN);
   disp( accuracy);

    % 计算F1分数 
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1_score = 2 * precision * recall / (precision + recall);
    disp( f1_score);

    % 计算MCC分数
    mcc_numerator = TP * TN - FP * FN;
    mcc_denominator = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
    if mcc_denominator == 0
        mcc_score = 0;
    else
        mcc_score = mcc_numerator / mcc_denominator;
    end
    disp( mcc_score);

    % 计算AUC
    for k = 1:m*n-floor(pp/10)*9
        tp = 0;
        for t = 1:pp
            if position(1, t) <= k
                tp = tp + 1;
            end
        end
        tpr(1, k) = tp / pp;
        if k < m*n-pp+floor(pp/10)+1
            fp = k * pp - tp;
        else
            fp = floor(pp/10)*9*(m*n-pp+floor(pp/10)) + (pp-floor(pp/10)*9)*k - tp;
        end
        fpr(1, k) = fp / (floor(pp/10)*9*(m*n-pp+floor(pp/10)-1) + (pp-floor(pp/10)*9)*(m*n-floor(pp/10)*9-1));
    end

    clear area;
    area(1, 1) = tpr(1, 1) * fpr(1, 1) / 2;
    for k = 2:m*n-floor(pp/10)*9
        area(1, k) = [tpr(1, k-1) + tpr(1, k)] * [fpr(1, k) - fpr(1, k-1)] / 2;
    end
    overallauc = sum(area);
    disp( overallauc);
end
