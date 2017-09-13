function [net]=cnnbp(net,y)
%�ú�������ǰ�򴫲��������������з��򴫲���������ʧ������Ȩֵ��ƫ�õ�ƫ����
%net��ʾ�������磬�����Ǿ��������
%y��ʾѵ�����ݵ����������y=1xbatchsize����
%net��ʾӦ��bp�㷨������Ȩֵ����ƫ������õ�������

n=numel(net.layers);%����n=5

%% sigmoid�����
C=1;
net.e=net.o{1,1}-net.o{2,1};%net.e=[48xbatchsize],net.o{1,1}��ʵ�������net.o{2,1}���������
L=sqrt(sum(net.e.*net.e,1));
idx1=find(y==1);
idx2=find(y==0 & L<C);
idx3=find(y==0 & L>=C);
net_1=net.e(:,idx1);
net_2=net.e(:,idx2);
net_3=net.e(:,idx3);

net.L=(sum(L(idx1))+sum(C-L(idx2)))/size(net.e,2);%��ʧ����
%net.od=[48xbatchsize]
if(size(idx1,2)~=0)
    temp1=repmat(L(idx1),size(net.e,1),1);
    net.od(:,idx1)=net_1./temp1.*(net.o{1,1}(:,idx1).*(1-net.o{1,1}(:,idx1)));%�����������
end
if(size(idx2,2)~=0)
    temp2=repmat(L(idx2),size(net.e,1),1);
    net.od(:,idx2)=-net_2./temp2.*(net.o{1,1}(:,idx2).*(1-net.o{1,1}(:,idx2)));%�����������
end
net.od(:,idx3)=0;%�����������

%�в�򴫲���ǰһ��
%net.ffW��batchsize*192���󣬱�ʾ��n(5)������һ�������Ȩֵw
net.fvd=(net.ffW'*net.od);%net.fvd��192*48�ľ���
if(strcmp(net.layers{n}.type,'c'))%��������ڶ����Ǿ����,���ﲻ�Ǿ����
    net.fvd=net.fvd.*(net.fv{1,1}.*(1-net.fv{1,1}));%net.fv�ǵ����ڶ���ļ����������
end

%���°ѵ����ڶ����������ʽ�����Ȼ��߲в���ϳɾ�����ʽ����Ϊʵ���ϵ����ڶ�����Ǿ���
%��ʽ���²����㣬ÿ��ӳ��ͼ�Ĵ�С������4x4����12��ӳ��ͼ
sa=size(net.layers{n}.a{1,1});%�����ڶ���ÿ������ͼ�Ĵ�Сsa=[4 4 batchsize];
fvnum=sa(1)*sa(2);%fvnum=16
if(size(net.fvd,2)<=1)
    sa(3)=1;
end
for j=1:1:size(net.layers{n}.a,2)%j�ķ�Χ��1-12
    temp=net.fvd((j-1)*fvnum+1:j*fvnum,:);
    net.layers{n}.d{j}=reshape(temp,sa(1),sa(2),sa(3));%d���浹���ڶ��������Ȼ��߲в�
end

%�����ڶ���֮ǰ�Ĳ���������Ȼ��߲в�͵����ڶ���ķ�ʽ��һ��
for i=n-1:-1:1
    if(strcmp(net.layers{i}.type,'c'))%�����
        for j=1:1:size(net.layers{i}.a,2)%�ò�����ӳ��ĸ���
            %net.layers{i}.d{j}������ǵ�i���j������ӳ��������Ȼ��߲в�
            %expend�Ĳ����൱�ڶ�l+1�㣨�²����㣩�ĵ�j�������Ƚ����ϲ���
            temp_1=net.layers{i}.a{1,j}.*(1-net.layers{i}.a{1,j});
%             temp_2=expand(net.layers{i+1}.d{j},[net.layers{i+1}.scale,net.layers{i+1}.scale,1])/(net.layers{i+1}.scale^2);
            temp_2=expand(net.layers{i+1}.d{j},[net.layers{i+1}.scale,net.layers{i+1}.scale,1]);
            %���ھ���㣬��Ϊ���������sigmod���������������Ȼ��߲в���Ҫ�����������
            net.layers{i}.d{j}=temp_1.*temp_2;
        end
    elseif(strcmp(net.layers{i}.type,'s'))%�²�����
        for j=1:1:size(net.layers{i}.a,2)%�ò�����ӳ�����
            z=zeros(size(net.layers{i}.a{1,1}));
            for p=1:1:size(net.layers{i+1}.a,2)%��i+1������ӳ�����
                z=z+convn(net.layers{i+1}.d{p},rot180(net.layers{i+1}.k{j,p}),'full');
            end
            %�����²����㣬��Ϊ���û�о���sigmod���������������Ȼ��߲в��Ҫ�����������
            net.layers{i}.d{j}=z;
        end
    end
end

% �����ݶ�
%�����Ӳ�����û�в�����Ҳû�м�������������Ӳ�����û����Ҫ���Ĳ���
for i=2:1:n
    %���������Ҫ����������Ԫ��Ȩֵw��ƫ��b,��Ϊ�²������Ȩֵw�̶�Ϊ1/4,ƫ��b=0
    %���²������Ȩֵ��ƫ�ù̶���Ϊ�˼򻯼��㣬��������Ч��
    if(strcmp(net.layers{i}.type,'c'))%�����
        for j=1:1:size(net.layers{i}.a,2)%���������ӳ��ͼ�ĸ���
            for p=1:1:size(net.layers{i-1}.a,2)%�����ǰһ������ӳ��ͼ����
                %����������size(net.layers{i}.d{j},3)�����ǳ���batchsize=50,
                %��Ϊnet.layers{i-1}.a{p}��net.layers{i}.d{j}������50��ͼ��convn���
                %�Ѷ�Ӧ��50�������ӣ������Ҫ��һ����ֵ
                %����ʹ��flipall��net.layers{i-1}.a{p}�ĵ���άҲ��ת�ˣ�����Ϊ��ʹ��convn��ʱ��
                % ��net.layers{i}.d{j}����άҲ��ת�ˣ����Ϊ��ʹ50��patch��Ӧ����Ҫ��net.layers{i-1}.a{p}
                %��50��patch�ĵ���άҲ������ת
                 net.layers{i}.dk{p,j}=convn(flipall(net.layers{i-1}.a{1,p}),...
                     net.layers{i}.d{j},'valid')/size(net.layers{i}.d{j},3);
                
                %�ȼ��������ʵ��
                %net.layers{i}.dk{p}{j}=rot180(convn(net.layers{i-1}.a{1,p},...
                    %flipall(net.layers{i}.d{j}),'valid'))/size(net.layers{i}.d{j},3);
            end
            %�������size(net.layers{i}.d{j},3)ͬ������Ϊ����batchsize=50,
            net.layers{i}.db{j}=sum(net.layers{i}.d{j}(:))/size(net.layers{i}.d{j},3);
        end
    end
end

%�����ڶ���͵�����һ��֮����ݶȱ仯��ƫ�ñ仯
net.dffW=net.od*net.fv{1,1}'/size(net.od,2);%net.od=10x50����net.fv=192x50����net.dffW=10x192����
net.dffb=mean(net.od,2);%���з������ֵ

end


























