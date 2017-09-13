function net=cnnsetup(net,width,height)
%该函数的作用是设置卷积神经网络的参数
%net表示需要设置的网络，这里指的是cnn
%size表示二维图像块的大小，默认是28

assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ...
 ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' myOctaveVersion]);
inputmaps=1;%输入层个数是1，这里指输入的一副测试数据
mapsize=[width,height];%得到训练样本每幅图像的行数和列数

%n=numl(A)返回数组A中元素个数
%net.layers中共有5个struct类型的元素
for i=1:1:numel(net.layers)%i的最大值是5
    if(strcmp(net.layers{i}.type,'s'))%如果是子采样层
        mapsize=mapsize/net.layers{i}.scale;%该子采样层大小是上层的1/2
        assert(all(floor(mapsize)==mapsize), ...
        ['层 ' num2str(i) ' 大小必须是整数. Actual: ' num2str(mapsize)]);
        for j=1:1:inputmaps%inputmaps表示上一层特征图个数
            net.layers{i}.b{j}=0;%将偏置初始化为0,下采样层不设置权值w,这里仅仅初始化了偏置
        end
    end
    
    if(strcmp(net.layers{i}.type,'c'))%如果是卷积层
        mapsize=mapsize-net.layers{i}.kernelsize+1;%本卷积层的大小,即mapsize保存本层特征图大小
        %fan_out保存卷积层卷积核权值w,偏置b在下面单独保存
        fan_out=net.layers{i}.outputmaps*net.layers{i}.kernelsize^2;
        for p=1:1:net.layers{i}.outputmaps %卷积层输出特征图个数
            fan_in=inputmaps*net.layers{i}.kernelsize^2;
            %卷积层输入特征图个数
            for q=1:1:inputmaps
                net.layers{i}.k{q,p}=(rand(net.layers{i}.kernelsize)-0.5)*2* ...
                    sqrt(6/(fan_in+fan_out));%初始化卷积层权值w
                net.layers{i}.vk{q,p}=zeros(size(net.layers{i}.k{q,p}));%动量项
            end
            net.layers{i}.b{p}=0;%初始化卷积层偏置b
        end
        %下一层的输入特征图个数是本层输出的特征图的个数
        inputmaps=net.layers{i}.outputmaps;
    end
end

%此时mapsize表示最后一个下采样层的大小，inputmaps表示最后一个下采样层的特征图个数
fvnum=prod(mapsize)*inputmaps;%prod表示向量元素相乘,此时mapsize=[4,4],inputmaps=12
onum=net.out_nums;%最后一层输出神经元个数，此时onum=48
net.ffb=zeros(onum,1);%最后一层的偏置,net.ffb=48x1矩阵
%最后一层和倒数第二层的权值连接，这里是全连接的，net.ffW=48x192矩阵
net.ffW=(rand(onum,fvnum)-0.5)*2*sqrt(6/(onum+fvnum));
net.vffW=zeros(size(net.ffW));%动量项

end


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    