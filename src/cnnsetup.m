function net=cnnsetup(net,width,height)
%�ú��������������þ��������Ĳ���
%net��ʾ��Ҫ���õ����磬����ָ����cnn
%size��ʾ��άͼ���Ĵ�С��Ĭ����28

assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ...
 ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' myOctaveVersion]);
inputmaps=1;%����������1������ָ�����һ����������
mapsize=[width,height];%�õ�ѵ������ÿ��ͼ�������������

%n=numl(A)��������A��Ԫ�ظ���
%net.layers�й���5��struct���͵�Ԫ��
for i=1:1:numel(net.layers)%i�����ֵ��5
    if(strcmp(net.layers{i}.type,'s'))%������Ӳ�����
        mapsize=mapsize/net.layers{i}.scale;%���Ӳ������С���ϲ��1/2
        assert(all(floor(mapsize)==mapsize), ...
        ['�� ' num2str(i) ' ��С����������. Actual: ' num2str(mapsize)]);
        for j=1:1:inputmaps%inputmaps��ʾ��һ������ͼ����
            net.layers{i}.b{j}=0;%��ƫ�ó�ʼ��Ϊ0,�²����㲻����Ȩֵw,���������ʼ����ƫ��
        end
    end
    
    if(strcmp(net.layers{i}.type,'c'))%����Ǿ����
        mapsize=mapsize-net.layers{i}.kernelsize+1;%�������Ĵ�С,��mapsize���汾������ͼ��С
        %fan_out������������Ȩֵw,ƫ��b�����浥������
        fan_out=net.layers{i}.outputmaps*net.layers{i}.kernelsize^2;
        for p=1:1:net.layers{i}.outputmaps %������������ͼ����
            fan_in=inputmaps*net.layers{i}.kernelsize^2;
            %�������������ͼ����
            for q=1:1:inputmaps
                net.layers{i}.k{q,p}=(rand(net.layers{i}.kernelsize)-0.5)*2* ...
                    sqrt(6/(fan_in+fan_out));%��ʼ�������Ȩֵw
                net.layers{i}.vk{q,p}=zeros(size(net.layers{i}.k{q,p}));%������
            end
            net.layers{i}.b{p}=0;%��ʼ�������ƫ��b
        end
        %��һ�����������ͼ�����Ǳ������������ͼ�ĸ���
        inputmaps=net.layers{i}.outputmaps;
    end
end

%��ʱmapsize��ʾ���һ���²�����Ĵ�С��inputmaps��ʾ���һ���²����������ͼ����
fvnum=prod(mapsize)*inputmaps;%prod��ʾ����Ԫ�����,��ʱmapsize=[4,4],inputmaps=12
onum=net.out_nums;%���һ�������Ԫ��������ʱonum=48
net.ffb=zeros(onum,1);%���һ���ƫ��,net.ffb=48x1����
%���һ��͵����ڶ����Ȩֵ���ӣ�������ȫ���ӵģ�net.ffW=48x192����
net.ffW=(rand(onum,fvnum)-0.5)*2*sqrt(6/(onum+fvnum));
net.vffW=zeros(size(net.ffW));%������

end


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    