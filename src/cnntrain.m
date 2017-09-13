function [net]=cnntrain(net,x,y,opts,test_x,test_y)
%该函数完成卷积神经网络的训练
%net表示输入的网络，这里指的是卷积神经网络cnn
%x表示网络输入的训练数据，[28x28x79100x2]
%y表示网络输出的训练数据，[1x79100]

net.rL=[];%保存训练损失函数
net.epochs_error=[];%每次迭代测试误差

for i=1:1:opts.numepochs%迭代次数循环
    disp(['epoch:',num2str(i),'/',num2str(opts.numepochs)]);
    tic;%开始计时
    m=size(y,2);%m=79100
    kk=randperm(m);%返回[1,m]之间所有整数的一个随机序列
    nums=floor(m/net.SIZE);%每次迭代划分为nums次更小的迭代
    for k=1:1:nums
        train_x=double(x(:,:,kk((k-1)*net.SIZE+1:k*net.SIZE),:))/255;
        train_y=double(y(1,kk((k-1)*net.SIZE+1:k*net.SIZE)));
        numbatches=net.SIZE/opts.batchsize;%net.SIZE=2000
        if(rem(numbatches,1)~=0)%rem(1.5,1)=0.5,因此rem是计算余数操作
            error('numbatches not integer!');
        end
        tt=randperm(net.SIZE);
        for j=1:1:numbatches
            temp_num=(j-1)*opts.batchsize+1:j*opts.batchsize;
            batch_x=train_x(:,:,tt(temp_num),:);%取出此时的opts.batchsize幅测试图像输入,batch_x=28x28xbatchsize
            batch_y=train_y(1,tt(temp_num));

            %在当前网络权值和网络输入下计算网络输出
            net=cnnff(net,batch_x);%前向传播,batch_x=28x28xbatchsize
            %得到上面的网络输出后，通过对应的样本标签用bp算法得到误差对网络权值的偏导数
            net=cnnbp(net,batch_y);%反向传播,batch_y=1xbatchsize
            %得到误差对权值的导数后，就通过权值更新方法更新权值
            net=cnnapplygrads(net,opts);

            if(isempty(net.rL))
                net.rL(1)=net.L;
            end
            net.rL(end+1)=0.99*net.rL(end)+0.01*net.L;%保存历史的误差值，以便绘图
        end
    end
    
    % 训练剩下的数据
    train_x=double(x(:,:,kk(nums*net.SIZE+1:m),:))/255;
    train_y=double(y(1,kk(nums*net.SIZE+1:m)));
    numbatches=floor(size(train_y,2)/opts.batchsize);
    tt=randperm(size(train_y,2));
    for j=1:1:numbatches
        temp_num=(j-1)*opts.batchsize+1:j*opts.batchsize;
        batch_x=train_x(:,:,tt(temp_num),:);
        batch_y=train_y(1,tt(temp_num));

        net=cnnff(net,batch_x);
        net=cnnbp(net,batch_y);
        net=cnnapplygrads(net,opts);

        if(isempty(net.rL))
            net.rL(1)=net.L;
        end
        net.rL(end+1)=0.99*net.rL(end)+0.01*net.L;%保存历史的误差值，以便绘图
    end
               
    fprintf('第%d次迭代花费时间是：%.3f秒。\n',i,toc);%精确地毫秒
    %每次迭代后都计算出分类误差，并保存下来
    [er,~,~,~,~,~,~]=cnntest(net,test_x,test_y);
    net.epochs_error=[net.epochs_error,er];%保存每次迭代后计算
end

end


















