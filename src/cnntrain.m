function [net]=cnntrain(net,x,y,opts,test_x,test_y)
%�ú�����ɾ���������ѵ��
%net��ʾ��������磬����ָ���Ǿ��������cnn
%x��ʾ���������ѵ�����ݣ�[28x28x79100x2]
%y��ʾ���������ѵ�����ݣ�[1x79100]

net.rL=[];%����ѵ����ʧ����
net.epochs_error=[];%ÿ�ε����������

for i=1:1:opts.numepochs%��������ѭ��
    disp(['epoch:',num2str(i),'/',num2str(opts.numepochs)]);
    tic;%��ʼ��ʱ
    m=size(y,2);%m=79100
    kk=randperm(m);%����[1,m]֮������������һ���������
    nums=floor(m/net.SIZE);%ÿ�ε�������Ϊnums�θ�С�ĵ���
    for k=1:1:nums
        train_x=double(x(:,:,kk((k-1)*net.SIZE+1:k*net.SIZE),:))/255;
        train_y=double(y(1,kk((k-1)*net.SIZE+1:k*net.SIZE)));
        numbatches=net.SIZE/opts.batchsize;%net.SIZE=2000
        if(rem(numbatches,1)~=0)%rem(1.5,1)=0.5,���rem�Ǽ�����������
            error('numbatches not integer!');
        end
        tt=randperm(net.SIZE);
        for j=1:1:numbatches
            temp_num=(j-1)*opts.batchsize+1:j*opts.batchsize;
            batch_x=train_x(:,:,tt(temp_num),:);%ȡ����ʱ��opts.batchsize������ͼ������,batch_x=28x28xbatchsize
            batch_y=train_y(1,tt(temp_num));

            %�ڵ�ǰ����Ȩֵ�����������¼����������
            net=cnnff(net,batch_x);%ǰ�򴫲�,batch_x=28x28xbatchsize
            %�õ���������������ͨ����Ӧ��������ǩ��bp�㷨�õ���������Ȩֵ��ƫ����
            net=cnnbp(net,batch_y);%���򴫲�,batch_y=1xbatchsize
            %�õ�����Ȩֵ�ĵ����󣬾�ͨ��Ȩֵ���·�������Ȩֵ
            net=cnnapplygrads(net,opts);

            if(isempty(net.rL))
                net.rL(1)=net.L;
            end
            net.rL(end+1)=0.99*net.rL(end)+0.01*net.L;%������ʷ�����ֵ���Ա��ͼ
        end
    end
    
    % ѵ��ʣ�µ�����
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
        net.rL(end+1)=0.99*net.rL(end)+0.01*net.L;%������ʷ�����ֵ���Ա��ͼ
    end
               
    fprintf('��%d�ε�������ʱ���ǣ�%.3f�롣\n',i,toc);%��ȷ�غ���
    %ÿ�ε����󶼼��������������������
    [er,~,~,~,~,~,~]=cnntest(net,test_x,test_y);
    net.epochs_error=[net.epochs_error,er];%����ÿ�ε��������
end

end


















