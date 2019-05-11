close all;
clear all;

%% 读取训练数据集，并保存，为接下来生成训练需要的图像对做准备
M=28;%图像大小，需要手动设置
p = genpath('..\AR人脸数据库处理后的数据集\训练集');%根据需要手动修改路径
length_p = size(p,2);%字符串长度
path={};
temp=[];
for i=1:1:length_p%寻找分隔符；
    if(p(i)~=';')
        temp=[temp p(i)];
    else
        temp=[temp '\'];
        path=[path;temp];
        temp=[];
    end
end

%读取各个文件夹中的图像
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
            face_image_train(:,:,k)=image;%如果image是RGB图像，需要先转换为灰度图像
        end
    end
end
face_image_train=uint8(face_image_train);
save AR_face_image_train.mat face_image_train%%保存训练数据集



%% 读取训测试据集，并保存，为接下来生成测试需要的图像对做准备
M=28;%图像大小，需要手动设置
p = genpath('..\AR人脸数据库处理后的数据集\测试集');%根据需要手动修改路径
length_p = size(p,2);%字符串长度
path={};
temp=[];
for i=1:1:length_p%寻找分隔符；
    if(p(i)~=';')
        temp=[temp p(i)];
    else
        temp=[temp '\'];
        path=[path;temp];
        temp=[];
    end
end

%读取各个文件夹中的图像
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
            face_image_test(:,:,k)=image;%如果image是RGB图像，需要先转换为灰度图像
        end
    end
end

face_image_test=uint8(face_image_test);
save AR_face_image_test.mat face_image_test%%保存测试数据集



    















        
