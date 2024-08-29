clc
clear
orderC=2.^[0:15];
orderr=2.^[-15:0];
bestsen=0;
bestspe=0;
bestAUC=0;
bestGm=0;
bestg=0;
bestC=0;
bestF1=0;
bestRecall=0;
bestPrecision=0;
bestAccuracy1=0;
bestsen_std=0;
bestspe_std=0; 
bestAUC_std=0;
bestGm_std=0;
bestRecall_std=0;
bestPrecision_std=0;
bestAccuracy1_std=0;
bestF1_std=0;
for C=orderC
for r=orderr 
Gm=[];
AUC=[];
Spe=[];
Sen=[];
Precision=[];
Recall=[];
Accuracy1=[];
F1=[];
for p=0:9 
load('po.mat')
load('ne.mat')
L=size(train_negative,1)/size(train_positive,1);
cmd = ['-c ', num2str(C), ' -g ', num2str(r)];
rand('state',p);
tot_num_train_positive=size(train_positive,1);
tot_num_train_negative=size(train_negative,1);
randomordertrain=randperm(tot_num_train_positive);
randomordertrain1=randperm(tot_num_train_negative);
train_positive=train_positive(randomordertrain, :);
train_negative=train_negative(randomordertrain1, :);
b=num2str(tot_num_train_positive);
N=length(b);
units=b(N);
piece=(tot_num_train_positive-str2num(units))/10;    
bb=num2str(tot_num_train_negative);
NN=length(bb);
units=bb(NN);
piece1=(tot_num_train_negative-str2num(units))/10;
for i=1:9
c(1,i)={train_positive(1+piece*(i-1):piece*i,:)};
end
c(1,10)={train_positive(1+piece*i:end,:)};
for i=1:9
cc(1,i)={train_negative(1+piece1*(i-1):piece1*i,:)};
end
cc(1,10)={train_negative(1+piece1*i:end,:)};
 Pre=[];
 Trainlabel=[];
for i=1:10
test_positive=c{1,1};
train_positive=[c{1,2};c{1,3};c{1,4};c{1,5};c{1,6};c{1,7};c{1,8};c{1,9};c{1,10}]; 
c{1,11}=c{1,1};
for j=1:10
c{1,j}=c{1,j+1};
end
test_negative=cc{1,1};
train_negative=[cc{1,2};cc{1,3};cc{1,4};cc{1,5};cc{1,6};cc{1,7};cc{1,8};cc{1,9};cc{1,10}];
cc{1,11}=cc{1,1};
for j=1:10
cc{1,j}=cc{1,j+1};
end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  U=train_negative;
  P=train_positive;
  U1=U(:,1:end-2);
  P1=P(:,1:end-2);
  traindata1=[P1;U1];
  L1=ones(size(P1,1),1);    
  L2=-ones(size(U1,1),1);
  Center_pos=sum(P1)/size(P1,1);
  Center_neg=sum(U1)/size(U1,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
guanliandu_po=P(:,end-1);
guanliandu_ne=U(:,end-1);
A11=P(:,end);
A22=U(:,end);
test_positive=test_positive(:,1:end-2);
test_negative=test_negative(:,1:end-2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for i=1:size(P1,1)
   W1(i,1)=guanliandu_po(i,:);
  end
  for i=1:size(U1,1)
     W2(i,1)=guanliandu_ne(i,:);
  end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(train_positive, 1)
V1(i, 1) = (1- W1(i,:))*A11(i,:);
end
for i = 1:size(train_negative, 1)
V2(i, 1) = (1- W2(i,:))*A22(i,:);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(train_positive, 1)
if  W1(i, 1)< V1(i, 1) 
    WW1(i,1)=0;
else 
    WW1(i,1) =(1-V1(i, :))/(2-V1(i, 1)-W1(i, :));
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(train_negative, 1)
if  W2(i, 1)< V2(i, 1) 
    WW2(i,1)=0;
else 
    WW2(i,1) =(1/L)*(((1-V2(i, :))/(2-V2(i, 1)-W2(i, :))));
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W=[W1; WW2];
clear dis1 dis2 W1 WW2
  train_label1=[L1;L2];    
  testdata=[test_positive;test_negative];
  test_label=[ones(size(test_positive,1),1); -ones(size(test_negative,1),1)];
  model = svmtrain(W,train_label1, traindata1, cmd); 
    [predict_label1, accuracy1,dec_values1] = svmpredict(test_label, testdata, model);
  %%%%%%%%%%%%%%%%%%%%
  Pre=[Pre;dec_values1];
  Trainlabel=[Trainlabel; test_label];
  
end
[auc, curve] = roc(Pre,Trainlabel, 1, -1);
[sen spe precision acc mcc recall F1_score gm]=performance(Pre,Trainlabel);
Gm=[Gm gm];
AUC=[AUC auc];
Sen=[Sen sen];
Spe=[Spe spe];
Precision=[Precision precision];
F1 = [F1 F1_score;];
Recall=[Recall recall];
Accuracy1=[Accuracy1 acc];
end
Gm_avg=sum(Gm)/10;
Gm_std=std(Gm);
AUC_avg=sum(AUC)/10;
AUC_std=std(AUC);
Sen_avg=sum(Sen)/10;
Sen_std=std(Sen);
Spe_avg=sum(Spe)/10;
Spe_std=std(Spe);
Precision_avg=sum(Precision)/10;
Precision_std=std(Precision);
Recall_avg=sum(Recall)/10;
Recall_std=std(Recall);
Accuracy1_avg=sum(Accuracy1)/10;
Accuracy1_std=std(Accuracy1);
F1_avg=sum(F1)/10;
F1_std=std(F1);
if (AUC_avg> bestAUC)
            bestGm = Gm_avg;bestAUC = AUC_avg; bestF1=F1_avg;bestC = C; bestg=r; bestsen=Sen_avg; bestspe=Spe_avg; bestPrecision = Precision_avg;bestRecall = Recall_avg;bestAccuracy1 = Accuracy1_avg;
            bestGm_std=Gm_std;bestsen_std=Sen_std; bestspe_std=Spe_std; bestAUC_std=AUC_std; bestPrecision_std=Precision_std; bestRecall_std=Recall_std; bestAccuracy1_std=Accuracy1_std;bestF1_std=F1_std;
end 
F1
C
r
bestGm
bestAUC
bestC
bestg
bestF1
bestPrecision
bestRecall
bestAccuracy1
bestsen
bestspe
end
end
Mean=[bestGm bestC bestg bestsen bestspe bestAUC bestPrecision bestRecall  bestAccuracy1 bestF1]
Standard_deviation=[bestC bestg bestGm_std bestsen_std bestspe_std bestAUC_std bestPrecision_std bestRecall_std bestAccuracy1_std bestF1_std]
result_all=[Mean;Standard_deviation];
save result_all result_all