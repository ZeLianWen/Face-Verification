function [net]=cnnbp(net,y)
%该函数根据前向传播结果，对网络进行反向传播，计算损失函数对权值和偏置的偏导数
%net表示输入网络，这里是卷积神经网络
%y表示训练数据的期望输出，y=1xbatchsize矩阵
%net表示应用bp算法对网络权值计算偏导数后得到的网络

n=numel(net.layers);%这里n=5

%% sigmoid输出层
C=1;
net.e=net.o{1,1}-net.o{2,1};%net.e=[48xbatchsize],net.o{1,1}是实际输出，net.o{2,1}是理想输出
L=sqrt(sum(net.e.*net.e,1));
idx1=find(y==1);
idx2=find(y==0 & L<C);
idx3=find(y==0 & L>=C);
net_1=net.e(:,idx1);
net_2=net.e(:,idx2);
net_3=net.e(:,idx3);

net.L=(sum(L(idx1))+sum(C-L(idx2)))/size(net.e,2);%损失函数
%net.od=[48xbatchsize]
if(size(idx1,2)~=0)
    temp1=repmat(L(idx1),size(net.e,1),1);
    net.od(:,idx1)=net_1./temp1.*(net.o{1,1}(:,idx1).*(1-net.o{1,1}(:,idx1)));%输出层灵敏度
end
if(size(idx2,2)~=0)
    temp2=repmat(L(idx2),size(net.e,1),1);
    net.od(:,idx2)=-net_2./temp2.*(net.o{1,1}(:,idx2).*(1-net.o{1,1}(:,idx2)));%输出层灵敏度
end
net.od(:,idx3)=0;%输出层灵敏度

%残差反向传播到前一层
%net.ffW是batchsize*192矩阵，表示第n(5)层和最后一层的连接权值w
net.fvd=(net.ffW'*net.od);%net.fvd是192*48的矩阵
if(strcmp(net.layers{n}.type,'c'))%如果倒数第二层是卷积层,这里不是卷积层
    net.fvd=net.fvd.*(net.fv{1,1}.*(1-net.fv{1,1}));%net.fv是倒数第二层的激活输出向量
end

%重新把倒数第二层的向量形式灵敏度或者残差组合成矩阵形式，因为实际上倒数第二层就是矩阵
%形式的下采样层，每个映射图的大小这里是4x4，共12个映射图
sa=size(net.layers{n}.a{1,1});%倒数第二层每个特征图的大小sa=[4 4 batchsize];
fvnum=sa(1)*sa(2);%fvnum=16
if(size(net.fvd,2)<=1)
    sa(3)=1;
end
for j=1:1:size(net.layers{n}.a,2)%j的范围是1-12
    temp=net.fvd((j-1)*fvnum+1:j*fvnum,:);
    net.layers{n}.d{j}=reshape(temp,sa(1),sa(2),sa(3));%d保存倒数第二层灵敏度或者残差
end

%倒数第二层之前的层计算灵敏度或者残差和倒数第二层的方式不一样
for i=n-1:-1:1
    if(strcmp(net.layers{i}.type,'c'))%卷积层
        for j=1:1:size(net.layers{i}.a,2)%该层特征映射的个数
            %net.layers{i}.d{j}保存的是第i层第j个特征映射的灵敏度或者残差
            %expend的操作相当于对l+1层（下采样层）的第j个灵敏度进行上采样
            temp_1=net.layers{i}.a{1,j}.*(1-net.layers{i}.a{1,j});
%             temp_2=expand(net.layers{i+1}.d{j},[net.layers{i+1}.scale,net.layers{i+1}.scale,1])/(net.layers{i+1}.scale^2);
            temp_2=expand(net.layers{i+1}.d{j},[net.layers{i+1}.scale,net.layers{i+1}.scale,1]);
            %对于卷积层，因为输出经过了sigmod函数，所以灵敏度或者残差需要乘以输出导数
            net.layers{i}.d{j}=temp_1.*temp_2;
        end
    elseif(strcmp(net.layers{i}.type,'s'))%下采样层
        for j=1:1:size(net.layers{i}.a,2)%该层特征映射个数
            z=zeros(size(net.layers{i}.a{1,1}));
            for p=1:1:size(net.layers{i+1}.a,2)%第i+1层特征映射个数
                z=z+convn(net.layers{i+1}.d{p},rot180(net.layers{i+1}.k{j,p}),'full');
            end
            %对于下采样层，因为输出没有经过sigmod函数，所以灵敏度或者残差不需要乘以输出导数
            net.layers{i}.d{j}=z;
        end
    end
end

% 计算梯度
%这里子采样层没有参数，也没有激活函数，所以在子采样层没有需要求解的参数
for i=2:1:n
    %这里仅仅需要计算卷积层神经元的权值w和偏置b,因为下采样层的权值w固定为1/4,偏置b=0
    %把下采样层的权值和偏置固定是为了简化计算，提高运算的效率
    if(strcmp(net.layers{i}.type,'c'))%卷积层
        for j=1:1:size(net.layers{i}.a,2)%卷积层特征映射图的个数
            for p=1:1:size(net.layers{i-1}.a,2)%卷积层前一层特征映射图个数
                %这里结果除以size(net.layers{i}.d{j},3)，就是除以batchsize=50,
                %因为net.layers{i-1}.a{p}和net.layers{i}.d{j}都包含50幅图像，convn结果
                %把对应的50幅结果相加，因此需要求一个均值
                %这里使用flipall把net.layers{i-1}.a{p}的第三维也反转了，是因为在使用convn的时候
                % 把net.layers{i}.d{j}第三维也反转了，因此为了使50个patch对应，需要对net.layers{i-1}.a{p}
                %中50个patch的第三维也进行旋转
                 net.layers{i}.dk{p,j}=convn(flipall(net.layers{i-1}.a{1,p}),...
                     net.layers{i}.d{j},'valid')/size(net.layers{i}.d{j},3);
                
                %等价于下面的实现
                %net.layers{i}.dk{p}{j}=rot180(convn(net.layers{i-1}.a{1,p},...
                    %flipall(net.layers{i}.d{j}),'valid'))/size(net.layers{i}.d{j},3);
            end
            %这里除以size(net.layers{i}.d{j},3)同样是因为输入batchsize=50,
            net.layers{i}.db{j}=sum(net.layers{i}.d{j}(:))/size(net.layers{i}.d{j},3);
        end
    end
end

%倒数第二层和倒数第一层之间的梯度变化和偏置变化
net.dffW=net.od*net.fv{1,1}'/size(net.od,2);%net.od=10x50矩阵，net.fv=192x50矩阵，net.dffW=10x192矩阵
net.dffb=mean(net.od,2);%沿列方向求均值

end


























