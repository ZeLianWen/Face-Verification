function net=cnnapplygrads(net,opts)
%�ú�������Ȩֵ��ƫ��
%net��ʾ��������磬����ָ���Ǿ��������
%opts��ʾ����Ĳ���
%���net��ʾȨֵ���º������

for i=2:1:numel(net.layers)
    %��Ϊ�˳����У�������Ȩֵw��ƫ��b�ǿ��Ե��ڵģ��²������Ȩֵ��ƫ��b�ǹ̶���
    if(strcmp(net.layers{i}.type,'c'))%�����
        for j=1:1:size(net.layers{i}.a,2)
            for p=1:1:size(net.layers{i-1}.a,2)
                if(net.weightPenaltyL2>0)%�������Ȩ�سͷ���
                    dw=net.layers{i}.dk{p,j}+net.weightPenaltyL2*net.layers{i}.k{p,j};
                else
                    dw=net.layers{i}.dk{p,j};
                end
                
                dw=opts.alpha*dw;
                if(net.momentum>0)%������ڶ�����
                    net.layers{i}.vk{p,j}=net.momentum*net.layers{i}.vk{p,j}+dw;
                    dw=net.layers{i}.vk{p,j};
                end
                
                net.layers{i}.k{p,j}=net.layers{i}.k{p,j}-dw;%����Ȩֵ           
            end
            net.layers{i}.b{j}=net.layers{i}.b{j}-opts.alpha*net.layers{i}.db{j};%����ƫ��
        end
    end
end

%���µ����ڶ�������һ���Ȩֵw��ƫ��
if(net.weightPenaltyL2>0)
    dw=net.dffW+net.weightPenaltyL2*net.ffW;
else
    dw=net.dffW;
end
dw=dw*opts.alpha;
if(net.momentum>0)%���ڶ�����
    net.vffW=net.momentum*net.vffW+dw;
    dw=net.vffW;
end

net.ffW=net.ffW-dw;
net.ffb=net.ffb-opts.alpha*net.dffb;
end




















    
                    
