function net=cnnapplygrads(net,opts)
%该函数更新权值和偏置
%net表示输入的网络，这里指的是卷积神经网络
%opts表示网络的参数
%输出net表示权值更新后的网络

for i=2:1:numel(net.layers)
    %因为此程序中，卷积层的权值w和偏置b是可以调节的，下采样层的权值和偏置b是固定的
    if(strcmp(net.layers{i}.type,'c'))%卷积层
        for j=1:1:size(net.layers{i}.a,2)
            for p=1:1:size(net.layers{i-1}.a,2)
                if(net.weightPenaltyL2>0)%如果存在权重惩罚项
                    dw=net.layers{i}.dk{p,j}+net.weightPenaltyL2*net.layers{i}.k{p,j};
                else
                    dw=net.layers{i}.dk{p,j};
                end
                
                dw=opts.alpha*dw;
                if(net.momentum>0)%如果存在动量项
                    net.layers{i}.vk{p,j}=net.momentum*net.layers{i}.vk{p,j}+dw;
                    dw=net.layers{i}.vk{p,j};
                end
                
                net.layers{i}.k{p,j}=net.layers{i}.k{p,j}-dw;%更新权值           
            end
            net.layers{i}.b{j}=net.layers{i}.b{j}-opts.alpha*net.layers{i}.db{j};%更新偏置
        end
    end
end

%更新倒数第二层和最后一层的权值w和偏置
if(net.weightPenaltyL2>0)
    dw=net.dffW+net.weightPenaltyL2*net.ffW;
else
    dw=net.dffW;
end
dw=dw*opts.alpha;
if(net.momentum>0)%存在动量项
    net.vffW=net.momentum*net.vffW+dw;
    dw=net.vffW;
end

net.ffW=net.ffW-dw;
net.ffb=net.ffb-opts.alpha*net.dffb;
end




















    
                    
