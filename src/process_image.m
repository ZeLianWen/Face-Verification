close all;
clear all;

%% ��ȡѵ�����ݼ��������棬Ϊ����������ѵ����Ҫ��ͼ�����׼��
M=28;%ͼ���С����Ҫ�ֶ�����
p = genpath('..\AR�������ݿ⴦�������ݼ�\ѵ����');%������Ҫ�ֶ��޸�·��
length_p = size(p,2);%�ַ�������
path={};
temp=[];
for i=1:1:length_p%Ѱ�ҷָ�����
    if(p(i)~=';')
        temp=[temp p(i)];
    else
        temp=[temp '\'];
        path=[path;temp];
        temp=[];
    end
end

%��ȡ�����ļ����е�ͼ��
k=0;
face_image_train=zeros(M,M,100);

file_num=size(path,1);
for i=2:1:file_num
    file_path=path{i};
    image_path_list= dir(strcat(file_path,'*.pgm'));
    if(length(image_path_list)<=0)
        image_path_list= dir(strcat(file_path,'*.bmp'));
    end
    if(length(image_path_list)<=0)
        image_path_list= dir(strcat(file_path,'*.tif'));
    end
    if(length(image_path_list)<=0)
        image_path_list= dir(strcat(file_path,'*.jpg'));
    end
    if(length(image_path_list)<=0)
        image_path_list= dir(strcat(file_path,'*.png'));
    end
    img_num=length(image_path_list);

    if(img_num>0)
        for j=1:1:img_num
            k=k+1;
            image_name=image_path_list(j).name;
            image=imread(strcat(file_path,image_name));
            face_image_train(:,:,k)=image;%���image��RGBͼ����Ҫ��ת��Ϊ�Ҷ�ͼ��
        end
    end
end
face_image_train=uint8(face_image_train);
save AR_face_image_train.mat face_image_train%%����ѵ�����ݼ�



%% ��ȡѵ���Ծݼ��������棬Ϊ���������ɲ�����Ҫ��ͼ�����׼��
M=28;%ͼ���С����Ҫ�ֶ�����
p = genpath('..\AR�������ݿ⴦�������ݼ�\���Լ�');%������Ҫ�ֶ��޸�·��
length_p = size(p,2);%�ַ�������
path={};
temp=[];
for i=1:1:length_p%Ѱ�ҷָ�����
    if(p(i)~=';')
        temp=[temp p(i)];
    else
        temp=[temp '\'];
        path=[path;temp];
        temp=[];
    end
end

%��ȡ�����ļ����е�ͼ��
k=0;
face_image_test=zeros(M,M,100);

file_num=size(path,1);
for i=2:1:file_num
    file_path=path{i};
    image_path_list= dir(strcat(file_path,'*.pgm'));
    if(length(image_path_list)<=0)
        image_path_list= dir(strcat(file_path,'*.bmp'));
    end
    if(length(image_path_list)<=0)
        image_path_list= dir(strcat(file_path,'*.tif'));
    end
    if(length(image_path_list)<=0)
        image_path_list= dir(strcat(file_path,'*.jpg'));
    end
    if(length(image_path_list)<=0)
        image_path_list= dir(strcat(file_path,'*.png'));
    end
    img_num=length(image_path_list);

    if(img_num>0)
        for j=1:1:img_num
            k=k+1;
            image_name=image_path_list(j).name;
            image=imread(strcat(file_path,image_name));
            face_image_test(:,:,k)=image;%���image��RGBͼ����Ҫ��ת��Ϊ�Ҷ�ͼ��
        end
    end
end

face_image_test=uint8(face_image_test);
save AR_face_image_test.mat face_image_test%%����������ݼ�



    















        
