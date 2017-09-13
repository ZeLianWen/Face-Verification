function net=cnnff(net,x)
%该函数根据当前网络权值和网络输入，计算网络输出
%net表示输入的网络，这里表示卷积神经网络
%x表示网络的输入，x=28x28x50x2
%net表示根据网络输入获得网络输出后的网络

n=numel(net.layers);%层数,n=5
net.layers{1}.a{1,1}=x(:,:,:,1);%网络的第一层就是输入，28x28x50
net.layers{1}.a{2,1}=x(:,:,:,2);
net.fv=cell(2,1);

for kk=1:1:2%输入图像是两幅
    inputmaps=1;%输入层只有一个特征映射，也就是原始输入图像
    for i=2:1:n%对于每一层
        if(strcmp(net.layers{i}.type,'c'))%卷积层
            for p=1:1:net.layers{i}.outputmaps%对于卷积层的每一个输出映射
                temp_size=net.layers{i}.kernelsize-1;
                if(size(x,3)>1)
                    temp_size=[temp_size,temp_size,0];
                else
                    temp_size=[temp_size,temp_size];
                end
                z=zeros(size(net.layers{i-1}.a{kk,1})-temp_size);
                for q=1:1:inputmaps%对于每一个输入映射
                    %net.layers{i}.k{q}{p}是二维的，net.layers{i-1}.a{q}是三维的，z也是三维的
                    z=z+convn(net.layers{i-1}.a{kk,q},net.layers{i}.k{q,p},'valid');
                end
                %加上偏置，然后经过非线性函数
                net.layers{i}.a{kk,p}=sigm(z+net.layers{i}.b{p});
            end
            inputmaps=net.layers{i}.outputmaps;
        elseif(strcmp(net.layers{i}.type,'s'))%下采样层
            for j=1:1:inputmaps
                %均值滤波核，下采样层固定权值w=1/4,下采样层偏置固定为0
                temp_conv=ones(net.layers{i}.scale)/(net.layers{i}.scale^2);
                z=convn(net.layers{i-1}.a{kk,j},temp_conv,'valid');
                %每隔两个像素取值一个
                net.layers{i}.a{kk,j}=z(1:net.layers{i}.scale:end,1:net.layers{i}.scale:end,:);
            end
        end
    end

    %映射倒数第二层的特征图为向量形式
    for j=1:1:size(net.layers{n}.a,2)%倒数第二层特征图个数
        sa=size(net.layers{n}.a{kk,j});%a{j}是三维矩阵，前两维是图像大小，第三维是图像个数,sa=4x4x50矩阵
        %将倒数第二层所有的特征映射拉成一条列向量，列索引对应样本，net.fv是倒数第二层的输出向量
        if(size(x,3)<=1)
            sa(3)=1;
        end
        net.fv{kk,1}=[net.fv{kk,1};reshape(net.layers{n}.a{kk,j},sa(1)*sa(2),sa(3))];%net.fv=192x50矩阵
    end

    %计算最后一层的输出值,net.ffW=10x192矩阵，net.fv=192x50矩阵，net.fb=10x1矩阵
    net.o{kk,1}=sigm(net.ffW*net.fv{kk,1}+repmat(net.ffb,1,size(net.fv{kk,1},2)));%%sigma输出
end

end
          
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


