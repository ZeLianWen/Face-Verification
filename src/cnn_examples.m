%%%%%%%%%% 该程序使用卷积神经网络运行人脸认证任务%%%%%%%%%%%%

%清除窗口变量
close all
clear all

%% 加载训练和测试数据
load AR_face_data_train;%仅仅包括训练集合
load AR_face_data_test;%仅仅包含测试集合

test_x=double(test_x)/255;%一定记住转换为浮点类型数据
test_y=double(test_y);

%% 加载训练好的数据
use_pre=false;%use_pre=true表示加载之前训练好的模型继续训练，如果为false,则重新训练，如果只是验证测试数据集，则设置为false
old_rL=[];
old_epochs_error=[];
if(use_pre)
    load cnn
    old_rL=cnn.rL;
    old_epochs_error=cnn.epochs_error;
end

%% 构建卷积神经网络
opts.alpha=0.01;%学习率
opts.batchsize=50;%batch大小
opts.numepochs=205;%用同样的数据集训练的次数，也就是算法迭代次数，我们的模型已经训练了205次

%cnn是一个结构体，cnn.layers是5x1的cell,每个cell里面又是一个结构体（struct）
if(~use_pre)
    cnn.layers={
        struct('type','i'),...%输入层
        struct('type','c','outputmaps',6,'kernelsize',5),...%第一个卷积层
        struct('type','s','scale',2),...%第一个卷积层对应的下采样层
        struct('type','c','outputmaps',12,'kernelsize',5),...%第二个卷积层
        struct('type','s','scale',2)%第二个卷积层对应的下采样层
        };
    cnn.out_nums=48;%输出层节点数是48
    %每次取出训练集的2000个进行训练，因为如果训练集一次转换为浮点数，太占内存，
    %因此，每次取出一部分，这样即使很大的训练数据集，内存也不会耗尽，这一点很重要。
    cnn.SIZE=2000;
    %设置卷积神经网络
    width=size(train_x,2);%输入图像的宽度
    height=size(train_x,1);%输入图像的高度
    cnn=cnnsetup(cnn,width,height);
end

cnn.th=0.2;%距离阈值
cnn.momentum=0.9;%冲量项系数
cnn.weightPenaltyL2=0.0001;%权值惩罚

%% 使用训练样本训练卷积神经网络
cnn=cnntrain(cnn,train_x,train_y,opts,test_x,test_y);  
cnn.rL=[old_rL,cnn.rL];
cnn.epochs_error=[old_epochs_error,cnn.epochs_error];
save cnn.mat cnn%再次保存训练训练更新后的网络数据

%% 使用测试样本测试卷积神经网络
t1=clock;
[er,bad,dis,out,br,FPR,TPR]=cnntest(cnn,test_x,test_y);
t2=clock;
disp(['每对测试图像花费时间是：',num2str(etime(t2,t1)/size(test_y,2)),'秒']); 

%% 绘制损失函数曲线
if (exist('save_image','dir')==0)%如果文件夹不存在
    mkdir('save_image');
end
f1=figure;
plot(cnn.rL);
grid on
title('训练损失曲线','FontSize',16);
xlabel('网络更新次数','FontSize',16);
ylabel('训练损失','FontSize',16);
saveas(f1,'.\save_image\训练损失曲线.jpg');

f3=figure;
plot(cnn.rL(1:20000));
grid on
title('训练损失曲线','FontSize',16);
xlabel('网络更新次数','FontSize',16);
ylabel('训练损失','FontSize',16);
saveas(f3,'.\save_image\训练损失曲线2.jpg');

%% 绘制在测试集上的验证误差曲线
f2=figure;
plot(cnn.epochs_error);
grid on
title('测试误差率','FontSize',16);
xlabel('迭代次数','FontSize',16);
ylabel('测试误差率','FontSize',16);
saveas(f2,'.\save_image\测试误差率曲线.jpg');

%% 输出测试误差
fprintf('最后测试误差是：%.2f%%.\n',er*100);%显示小数点后两位数字
msgbox(['最后测试误差是：',num2str(er*100),'%'],'测试误差','none');
assert(er<0.12,'----------测试误差太大！----------');%如果误差大于0.12，会提示





































