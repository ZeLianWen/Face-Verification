%%%%%%%%%%%%%%%%%该模块根据训练好的模型计算ROC曲线，阈值从0.01-1，共100份%%%%%%%%%%%%%%%%%%%
close all;
clear all;

load cnn
load AR_face_data_test;
test_x=double(test_x)/255;%一定记住转换为浮点类型数据
test_y=double(test_y);

%获取测试集输出
cnn=cnnff(cnn,test_x);%计算输入x的输出

%输出归一化
o1=cnn.o{1,1};
pow_o1=sqrt(sum(o1.*o1,1));
pow_o1=repmat(pow_o1,size(o1,1),1);
o1=o1./pow_o1;

o2=cnn.o{2,1};
pow_o2=sqrt(sum(o2.*o2,1));
pow_o2=repmat(pow_o2,size(o1,1),1);
o2=o2./pow_o2;

%获取两个图像之间特征的欧几里得距离
diff=sqrt(sum((o1-o2).^2,1));

FPR=[];%False Positive Rate
TPR=[];%True Positive Rate
TH=[];
right_to_wrong=[];
wrong_to_right=[];

for th=0.01:0.01:1
    [idx_right_to_right]=find(diff<th & test_y==1);%实际上是正确匹配，输出为1的匹配对
    [idx_wrong_to_right]=find(diff<th & test_y==0);%实际上是错误匹配对，输出是1的匹配对
    [idx_right_to_wrong]=find(diff>=th & test_y==1);%实际上是正确匹配，输出是0的匹配对
    
    bad1=size(idx_right_to_wrong,2);
    bad2=size(idx_wrong_to_right,2);
    right_to_wrong=[right_to_wrong,bad1];
    wrong_to_right=[wrong_to_right,bad2];
    right1=size(idx_right_to_right,2);

    FPR=[FPR,bad2/size(find(test_y==0),2)];
    TPR=[TPR,right1/size(find(test_y==1),2)];
    TH=[TH,th];
end

%绘制曲线
f1=figure;
plot(FPR,TPR,'Color','red','LineWidth',2);
grid on
title('测试集ROC曲线','FontSize',16);
xlabel('False Positive Rate（FPR）','FontSize',16);
ylabel('True Positive Rate(TPR)','FontSize',16);
saveas(f1,'.\save_image\梦想的路_ROC.jpg');

%保存ROC曲线对应的数据
N=size(FPR,2);
fid = fopen('.\save_image\梦想的路_ROC.txt','wt');
for k=1:N;
fprintf(fid,'%.4f',TH(k));
fprintf(fid,'%c',' ');
fprintf(fid,'%.4f',TPR(k));
fprintf(fid,'%c',' ');
fprintf(fid,'%.4f\n',FPR(k));
end
fclose(fid);

%绘制阈值和wrong_to_right,right_to_wrong之间的曲线
f2=figure;
grid on
hold on
[AX,H1,H2]=plotyy(TH,wrong_to_right/size(find(test_y==0),2),...
                  TH,right_to_wrong/size(find(test_y==1),2));
xlabel('距离阈值(TH)','FontSize',16);

set(get(AX(1),'Ylabel'),'String','FP/(FP+TN)','FontSize',16);%设置Y轴
set(get(AX(2),'Ylabel'),'String','FN/(TP+FN)','FontSize',16);

set(H1,'Color','r','LineWidth',2);%设置线条属性
set(H2,'Color','b','LineWidth',2);

legend('Location','East','FP/(FP+TN)','FN/(TP+FN)');

set(AX(1),'Ycolor','r'); %设定两个Y轴的颜色为黑色
set(AX(2),'Ycolor','b'); %设定两个Y轴的颜色为黑色

text(0.5,0.8,'FP(假正例)，TN(真反例)','FontSize',14,'Color','r');
text(0.5,0.74,'TP(真正例)，FN(假反例)','FontSize',14,'Color','b');
text(0.22,0.064, '\leftarrow 交点对应的阈值作为最优阈值','FontSize',14);
saveas(f2,'.\save_image\距离阈值与FP和FN关系曲线.jpg');
hold off

%绘制阈值和整体错误率之间的关系
f3=figure;
plot(TH,(wrong_to_right+right_to_wrong)/size(test_y,2),'Color','red','LineWidth',2);
grid on
hold on
title('测试误差率与阈值的关系曲线','FontSize',16);
xlabel('距离阈值（TH）','FontSize',16);
ylabel('(FP+FN)/(FP+TN+TP+FN)','FontSize',16);
text(0.5,0.4,'FP(假正例)，TN(真反例)','FontSize',14,'Color','r');
text(0.5,0.34,'TP(真正例)，FN(假反例)','FontSize',14,'Color','b');
text(0.2,0.65,'测试误差率=(FP+FN)/(FP+TN+TP+FN)','FontSize',14,'Color','b');
text(0.2,0.063, '\leftarrow 最优阈值位置','FontSize',14);
hold off
saveas(f3,'.\save_image\测试误差率与阈值的关系曲线.jpg');
    

    
    
    
    
    
    
