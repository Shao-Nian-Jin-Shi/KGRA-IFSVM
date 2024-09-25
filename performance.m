% x: pre_label; y: test_label

function [sen spe precision acc mcc  recall F1_score gm]=performance(x,y)

x(x>0) = 1;x(x<=0) = 0;
y(y>0) = 1;y(y<=0) = 0;
xandy=x&y;
tp=sum(xandy(:)); % True positives
fp=sum(x(:))-tp ; % False positives
fn=sum(y(:))-tp ; % False negatives
tn=numel(xandy)-sum(x(:))-sum(y(:))+tp; % True negatives
sen=tp/(tp+fn);
spe=tn/(tn+fp);
precision=tp/(tp+fp);
acc=(tp+tn)/(tp+fp+tn+fn);
mcc=(tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));
recall = tp/(tp+fn);
F1_score= 2*(precision*recall)/(precision+recall);
gm=sqrt(sen*spe);

