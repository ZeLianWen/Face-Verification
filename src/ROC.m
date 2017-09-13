%%%%%%%%%%%%%%%%%��ģ�����ѵ���õ�ģ�ͼ���ROC���ߣ���ֵ��0.01-1����100��%%%%%%%%%%%%%%%%%%%
close all;
clear all;

load cnn
load AR_face_data_test;
test_x=double(test_x)/255;%һ����סת��Ϊ������������
test_y=double(test_y);

%��ȡ���Լ����
cnn=cnnff(cnn,test_x);%��������x�����

%�����һ��
o1=cnn.o{1,1};
pow_o1=sqrt(sum(o1.*o1,1));
pow_o1=repmat(pow_o1,size(o1,1),1);
o1=o1./pow_o1;

o2=cnn.o{2,1};
pow_o2=sqrt(sum(o2.*o2,1));
pow_o2=repmat(pow_o2,size(o1,1),1);
o2=o2./pow_o2;

%��ȡ����ͼ��֮��������ŷ����þ���
diff=sqrt(sum((o1-o2).^2,1));

FPR=[];%False Positive Rate
TPR=[];%True Positive Rate
TH=[];
right_to_wrong=[];
wrong_to_right=[];

for th=0.01:0.01:1
    [idx_right_to_right]=find(diff<th & test_y==1);%ʵ��������ȷƥ�䣬���Ϊ1��ƥ���
    [idx_wrong_to_right]=find(diff<th & test_y==0);%ʵ�����Ǵ���ƥ��ԣ������1��ƥ���
    [idx_right_to_wrong]=find(diff>=th & test_y==1);%ʵ��������ȷƥ�䣬�����0��ƥ���
    
    bad1=size(idx_right_to_wrong,2);
    bad2=size(idx_wrong_to_right,2);
    right_to_wrong=[right_to_wrong,bad1];
    wrong_to_right=[wrong_to_right,bad2];
    right1=size(idx_right_to_right,2);

    FPR=[FPR,bad2/size(find(test_y==0),2)];
    TPR=[TPR,right1/size(find(test_y==1),2)];
    TH=[TH,th];
end

%��������
f1=figure;
plot(FPR,TPR,'Color','red','LineWidth',2);
grid on
title('���Լ�ROC����','FontSize',16);
xlabel('False Positive Rate��FPR��','FontSize',16);
ylabel('True Positive Rate(TPR)','FontSize',16);
saveas(f1,'.\save_image\�����·_ROC.jpg');

%����ROC���߶�Ӧ������
N=size(FPR,2);
fid = fopen('.\save_image\�����·_ROC.txt','wt');
for k=1:N;
fprintf(fid,'%.4f',TH(k));
fprintf(fid,'%c',' ');
fprintf(fid,'%.4f',TPR(k));
fprintf(fid,'%c',' ');
fprintf(fid,'%.4f\n',FPR(k));
end
fclose(fid);

%������ֵ��wrong_to_right,right_to_wrong֮�������
f2=figure;
grid on
hold on
[AX,H1,H2]=plotyy(TH,wrong_to_right/size(find(test_y==0),2),...
                  TH,right_to_wrong/size(find(test_y==1),2));
xlabel('������ֵ(TH)','FontSize',16);

set(get(AX(1),'Ylabel'),'String','FP/(FP+TN)','FontSize',16);%����Y��
set(get(AX(2),'Ylabel'),'String','FN/(TP+FN)','FontSize',16);

set(H1,'Color','r','LineWidth',2);%������������
set(H2,'Color','b','LineWidth',2);

legend('Location','East','FP/(FP+TN)','FN/(TP+FN)');

set(AX(1),'Ycolor','r'); %�趨����Y�����ɫΪ��ɫ
set(AX(2),'Ycolor','b'); %�趨����Y�����ɫΪ��ɫ

text(0.5,0.8,'FP(������)��TN(�淴��)','FontSize',14,'Color','r');
text(0.5,0.74,'TP(������)��FN(�ٷ���)','FontSize',14,'Color','b');
text(0.22,0.064, '\leftarrow �����Ӧ����ֵ��Ϊ������ֵ','FontSize',14);
saveas(f2,'.\save_image\������ֵ��FP��FN��ϵ����.jpg');
hold off

%������ֵ�����������֮��Ĺ�ϵ
f3=figure;
plot(TH,(wrong_to_right+right_to_wrong)/size(test_y,2),'Color','red','LineWidth',2);
grid on
hold on
title('�������������ֵ�Ĺ�ϵ����','FontSize',16);
xlabel('������ֵ��TH��','FontSize',16);
ylabel('(FP+FN)/(FP+TN+TP+FN)','FontSize',16);
text(0.5,0.4,'FP(������)��TN(�淴��)','FontSize',14,'Color','r');
text(0.5,0.34,'TP(������)��FN(�ٷ���)','FontSize',14,'Color','b');
text(0.2,0.65,'���������=(FP+FN)/(FP+TN+TP+FN)','FontSize',14,'Color','b');
text(0.2,0.063, '\leftarrow ������ֵλ��','FontSize',14);
hold off
saveas(f3,'.\save_image\�������������ֵ�Ĺ�ϵ����.jpg');
    

    
    
    
    
    
    
