function net=cnnff(net,x)
%�ú������ݵ�ǰ����Ȩֵ���������룬�����������
%net��ʾ��������磬�����ʾ���������
%x��ʾ��������룬x=28x28x50x2
%net��ʾ����������������������������

n=numel(net.layers);%����,n=5
net.layers{1}.a{1,1}=x(:,:,:,1);%����ĵ�һ��������룬28x28x50
net.layers{1}.a{2,1}=x(:,:,:,2);
net.fv=cell(2,1);

for kk=1:1:2%����ͼ��������
    inputmaps=1;%�����ֻ��һ������ӳ�䣬Ҳ����ԭʼ����ͼ��
    for i=2:1:n%����ÿһ��
        if(strcmp(net.layers{i}.type,'c'))%�����
            for p=1:1:net.layers{i}.outputmaps%���ھ�����ÿһ�����ӳ��
                temp_size=net.layers{i}.kernelsize-1;
                if(size(x,3)>1)
                    temp_size=[temp_size,temp_size,0];
                else
                    temp_size=[temp_size,temp_size];
                end
                z=zeros(size(net.layers{i-1}.a{kk,1})-temp_size);
                for q=1:1:inputmaps%����ÿһ������ӳ��
                    %net.layers{i}.k{q}{p}�Ƕ�ά�ģ�net.layers{i-1}.a{q}����ά�ģ�zҲ����ά��
                    z=z+convn(net.layers{i-1}.a{kk,q},net.layers{i}.k{q,p},'valid');
                end
                %����ƫ�ã�Ȼ�󾭹������Ժ���
                net.layers{i}.a{kk,p}=sigm(z+net.layers{i}.b{p});
            end
            inputmaps=net.layers{i}.outputmaps;
        elseif(strcmp(net.layers{i}.type,'s'))%�²�����
            for j=1:1:inputmaps
                %��ֵ�˲��ˣ��²�����̶�Ȩֵw=1/4,�²�����ƫ�ù̶�Ϊ0
                temp_conv=ones(net.layers{i}.scale)/(net.layers{i}.scale^2);
                z=convn(net.layers{i-1}.a{kk,j},temp_conv,'valid');
                %ÿ����������ȡֵһ��
                net.layers{i}.a{kk,j}=z(1:net.layers{i}.scale:end,1:net.layers{i}.scale:end,:);
            end
        end
    end

    %ӳ�䵹���ڶ��������ͼΪ������ʽ
    for j=1:1:size(net.layers{n}.a,2)%�����ڶ�������ͼ����
        sa=size(net.layers{n}.a{kk,j});%a{j}����ά����ǰ��ά��ͼ���С������ά��ͼ�����,sa=4x4x50����
        %�������ڶ������е�����ӳ������һ������������������Ӧ������net.fv�ǵ����ڶ�����������
        if(size(x,3)<=1)
            sa(3)=1;
        end
        net.fv{kk,1}=[net.fv{kk,1};reshape(net.layers{n}.a{kk,j},sa(1)*sa(2),sa(3))];%net.fv=192x50����
    end

    %�������һ������ֵ,net.ffW=10x192����net.fv=192x50����net.fb=10x1����
    net.o{kk,1}=sigm(net.ffW*net.fv{kk,1}+repmat(net.ffb,1,size(net.fv{kk,1},2)));%%sigma���
end

end
          
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


