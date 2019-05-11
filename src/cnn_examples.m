%%%%%%%%%% �ó���ʹ�þ������������������֤����%%%%%%%%%%%%

%������ڱ���
close all
clear all

%% ����ѵ���Ͳ�������
load AR_face_data_train;%��������ѵ������
load AR_face_data_test;%�����������Լ���

test_x=double(test_x)/255;%һ����סת��Ϊ������������
test_y=double(test_y);

%% ����ѵ���õ�����
use_pre=false;%use_pre=true��ʾ����֮ǰѵ���õ�ģ�ͼ���ѵ�������Ϊfalse,������ѵ�������ֻ����֤�������ݼ���������Ϊfalse
old_rL=[];
old_epochs_error=[];
if(use_pre)
    load cnn
    old_rL=cnn.rL;
    old_epochs_error=cnn.epochs_error;
end

%% �������������
opts.alpha=0.01;%ѧϰ��
opts.batchsize=50;%batch��С
opts.numepochs=205;%��ͬ�������ݼ�ѵ���Ĵ�����Ҳ�����㷨�������������ǵ�ģ���Ѿ�ѵ����205��

%cnn��һ���ṹ�壬cnn.layers��5x1��cell,ÿ��cell��������һ���ṹ�壨struct��
if(~use_pre)
    cnn.layers={
        struct('type','i'),...%�����
        struct('type','c','outputmaps',6,'kernelsize',5),...%��һ�������
        struct('type','s','scale',2),...%��һ��������Ӧ���²�����
        struct('type','c','outputmaps',12,'kernelsize',5),...%�ڶ��������
        struct('type','s','scale',2)%�ڶ���������Ӧ���²�����
        };
    cnn.out_nums=48;%�����ڵ�����48
    %ÿ��ȡ��ѵ������2000������ѵ������Ϊ���ѵ����һ��ת��Ϊ��������̫ռ�ڴ棬
    %��ˣ�ÿ��ȡ��һ���֣�������ʹ�ܴ��ѵ�����ݼ����ڴ�Ҳ����ľ�����һ�����Ҫ��
    cnn.SIZE=2000;
    %���þ��������
    width=size(train_x,2);%����ͼ��Ŀ��
    height=size(train_x,1);%����ͼ��ĸ߶�
    cnn=cnnsetup(cnn,width,height);
end

cnn.th=0.2;%������ֵ
cnn.momentum=0.9;%������ϵ��
cnn.weightPenaltyL2=0.0001;%Ȩֵ�ͷ�

%% ʹ��ѵ������ѵ�����������
cnn=cnntrain(cnn,train_x,train_y,opts,test_x,test_y);  
cnn.rL=[old_rL,cnn.rL];
cnn.epochs_error=[old_epochs_error,cnn.epochs_error];
save cnn.mat cnn%�ٴα���ѵ��ѵ�����º����������

%% ʹ�ò����������Ծ��������
t1=clock;
[er,bad,dis,out,br,FPR,TPR]=cnntest(cnn,test_x,test_y);
t2=clock;
disp(['ÿ�Բ���ͼ�񻨷�ʱ���ǣ�',num2str(etime(t2,t1)/size(test_y,2)),'��']); 

%% ������ʧ��������
if (exist('save_image','dir')==0)%����ļ��в�����
    mkdir('save_image');
end
f1=figure;
plot(cnn.rL);
grid on
title('ѵ����ʧ����','FontSize',16);
xlabel('������´���','FontSize',16);
ylabel('ѵ����ʧ','FontSize',16);
saveas(f1,'.\save_image\ѵ����ʧ����.jpg');

f3=figure;
plot(cnn.rL(1:20000));
grid on
title('ѵ����ʧ����','FontSize',16);
xlabel('������´���','FontSize',16);
ylabel('ѵ����ʧ','FontSize',16);
saveas(f3,'.\save_image\ѵ����ʧ����2.jpg');

%% �����ڲ��Լ��ϵ���֤�������
f2=figure;
plot(cnn.epochs_error);
grid on
title('���������','FontSize',16);
xlabel('��������','FontSize',16);
ylabel('���������','FontSize',16);
saveas(f2,'.\save_image\�������������.jpg');

%% ����������
fprintf('����������ǣ�%.2f%%.\n',er*100);%��ʾС�������λ����
msgbox(['����������ǣ�',num2str(er*100),'%'],'�������','none');
assert(er<0.12,'----------�������̫��----------');%���������0.12������ʾ





































