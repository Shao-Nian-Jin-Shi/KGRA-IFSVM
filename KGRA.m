clc
clear
load('minmax_ecoli.mat')
labels = minmax_scaling(:, 1);
label_1_idx = (labels ==0); 
label_2_idx = (labels ==1); 
label_3_idx = (labels ==2); 
label_4_idx = (labels ==3); 
label_5_idx = (labels ==4); 
label_6_idx = (labels ==5); 
label_7_idx = (labels ==6); 
label_8_idx = (labels ==7); 
label_9_idx = (labels ==8); 
label_10_idx = (labels ==9);
%label_11_idx = (labels ==10);
%label_12_idx = (labels ==11);
%label_13_idx = (labels ==12);
%label_14_idx = (labels ==13);
%label_15_idx = (labels ==14);
%label_16_idx = (labels ==15);
%label_17_idx = (labels ==16);
%label_18_idx = (labels ==17);
%label_19_idx = (labels ==18);
%label_20_idx = (labels ==19);
%label_21_idx = (labels ==20);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train1 =minmax_scaling(label_1_idx, 2:end);
train2 =minmax_scaling(label_2_idx, 2:end); 
train3 =minmax_scaling(label_3_idx, 2:end);  
train4 =minmax_scaling(label_4_idx, 2:end);  
train5 =minmax_scaling(label_5_idx, 2:end); 
train6 =minmax_scaling(label_6_idx, 2:end); 
train7 =minmax_scaling(label_7_idx, 2:end);  
train8 =minmax_scaling(label_8_idx, 2:end);  
train9 =minmax_scaling(label_9_idx, 2:end);  
train10 =minmax_scaling(label_10_idx, 2:end); 
%train11 =minmax_scaling(label_11_idx, 2:end); 
%train12 =minmax_scaling(label_12_idx, 2:end);  
%train13 =minmax_scaling(label_13_idx, 2:end); 
%train14 =minmax_scaling(label_14_idx, 2:end); 
%train15 =minmax_scaling(label_15_idx, 2:end);  
%train16 =minmax_scaling(label_16_idx, 2:end);  
%train17 =minmax_scaling(label_17_idx, 2:end);  
%train18 =minmax_scaling(label_18_idx, 2:end);  
%train19 =minmax_scaling(label_19_idx, 2:end);  
%train20 =minmax_scaling(label_20_idx, 2:end);
train_positive=cat(1,train3);
train_negative=cat(1,train1,train2);
L=size(train_negative,1)/size(train_positive,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P=train_positive;
U=train_negative;
sigma = 1;
K_pos = zeros(size(P, 1), size(P, 1));
for i = 1:size(P, 1)
    for j = 1:size(P, 1)
        K_pos(i, j) = exp(-norm(P(i, :) - P(j, :))^2 / (2 * sigma^2));
    end
end
train_positive=K_pos*P;
K_neg = zeros(size(U, 1), size(U, 1));
for i = 1:size(U, 1)
    for j = 1:size(U, 1)
        K_neg(i, j) = exp(-norm(U(i, :) - U(j, :))^2 / (2 * sigma^2));
    end
end
train_negative=K_neg*U;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K=round(sqrt(length(train_positive)));
tree = KDTreeSearcher(train_positive);
[indices, distances] = knnsearch(tree, train_positive, 'K', K+1);
nearest_neighbors1 = indices(:, 2:end);
guanliandu1 = zeros(size(train_positive, 1), K);
for i = 1:size(train_positive, 1)
   nearest_neighbor_coords =train_positive(nearest_neighbors1(i, :), :);
   data_1 = abs(nearest_neighbor_coords- train_positive(i, :));
d_max=max(max(data_1));
d_min=min(min(data_1));
a=0.5;
data_1_1=(d_min+a*d_max)./(data_1+a*d_max);
xishu1(i, :)=mean(data_1_1,2);
disp(' k邻域之间的灰色关联度分别为：');
disp(xishu1);
guanliandu1=mean(xishu1,2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K=round(sqrt(length(train_negative)));
tree2 = KDTreeSearcher(train_negative);
[indices2, distances2] = knnsearch(tree2, train_negative, 'K', K+1);
nearest_neighbors2 = indices2(:, 2:end);
guanliandu2 = zeros(size(train_negative, 1), K);
for i = 1:size(train_negative, 1)
    nearest_neighbor_coords2 = train_negative(nearest_neighbors2(i, :), :);
    data_2 = abs(nearest_neighbor_coords2 - train_negative(i, :));
d_max2=max(max(data_2));
d_min2=min(min(data_2));
a=0.5;
data_2_2=(d_min2+a*d_max2)./(data_2+a*d_max2);
xishu2(i, :)=mean(data_2_2,2);
disp(' k邻域之间的灰色关联度分别为：');
disp(xishu2);
   guanliandu2=mean(xishu2,2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K=round(sqrt(length(train_negative)));
tree = KDTreeSearcher(train_negative);
[indices, distances] = knnsearch(tree, train_positive, 'K', K);
nearest_neighbors1 = indices(:, 1:end);
A11 = zeros(size(train_positive, 1), K);
for i = 1:size(train_positive, 1)
   nearest_neighbor_coords =train_negative(nearest_neighbors1(i, :), :);
   data_1 = abs(nearest_neighbor_coords- train_positive(i, :));
d_max=max(max(data_1));
d_min=min(min(data_1));
a=0.5;
data11=(d_min+a*d_max)./(data_1+a*d_max);
xishu11(i, :)=mean(data11,2);
disp(' k邻域之间的灰色关联度分别为：');
disp(xishu11);
A11=mean(xishu11,2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K=round(sqrt(length(train_positive)));
tree2 = KDTreeSearcher(train_positive);
[indices2, distances2] = knnsearch(tree2, train_negative, 'K', K);
nearest_neighbors2 = indices2(:, 1:end);
A22 = zeros(size(train_negative, 1), K);
for i = 1:size(train_negative, 1)
    nearest_neighbor_coords2 = train_positive(nearest_neighbors2(i, :), :);
    data_2 = abs(nearest_neighbor_coords2 - train_negative(i, :));
d_max2=max(max(data_2));
d_min2=min(min(data_2));
a=0.5; 
data22=(d_min2+a*d_max2)./(data_2+a*d_max2);
xishu22(i, :)=mean(data22,2);
disp(' k邻域之间的灰色关联度分别为：');
disp(xishu22);
A22=mean(xishu22,2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P (:,end+1)=guanliandu1;
U(:,end+1)=guanliandu2;
P(:,end+1)=A11;
U(:,end+1)=A22;
train_positive=P;
train_negative=U;