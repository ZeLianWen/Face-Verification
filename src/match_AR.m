close all;
clear all;

%% AR训练数据集
load AR_face_image_train;

N=size(face_image,3);
rows=size(face_image,1);
cols=size(face_image,2);
M=14;%AR数据集每个人14幅图像
per_AR=50;%每个人50幅不同的图像对，组成不匹配的图像集合

% 获得匹配的图像对，
match_images_AR=zeros(rows,cols,M*(M-1)*(N/M)/2,2);
match_labels_AR=zeros(1,M*(M-1)*(N/M)/2);
for i=1:1:N/M %对于每个人
    temp=0;
    for j=1:1:M-1 %对于一个人的M幅图像
        for k=j+1:1:M
            temp=temp+1;
            match_images_AR(:,:,(i-1)*(M*(M-1)/2)+temp,1)=face_image(:,:,(i-1)*M+j);
            match_images_AR(:,:,(i-1)*(M*(M-1)/2)+temp,2)=face_image(:,:,(i-1)*M+k);
            match_labels_AR(1,(i-1)*(M*(M-1)/2)+temp)=1;%1表示是匹配图像
        end
    end
end
match_images_AR=uint8(match_images_AR);
match_labels_AR=uint8(match_labels_AR);

% 获得不是正确匹配的图像对
no_match_images_AR=zeros(rows,cols,(N/M)*M*per_AR,2);
no_match_labels_AR=zeros(1,(N/M)*M*per_AR);
for i=1:1:N/M %对于每个人,AR训练数据集是100个人
    for j=1:1:M %对于一个人的M幅图像，M是14
        %产生从1-100之间的10个不等于j的互不相等的数字
        while(1)
            rands=ceil(rand(1,per_AR)*100);
            a=find(rands==i);
            if(size(a,2)==0);
                break;
            end
        end
        for k=1:1:per_AR
            rand_a=ceil(rand(1)*M);%产生1-M之间的随机数
            no_match_images_AR(:,:,(i-1)*M*per_AR+(j-1)*per_AR+k,1)=face_image(:,:,(i-1)*M+j);
            no_match_images_AR(:,:,(i-1)*M*per_AR+(j-1)*per_AR+k,2)=face_image(:,:,(rands(k)-1)*M+rand_a);
            no_match_labels_AR(1,(i-1)*M*per_AR+(j-1)*per_AR+k)=0;
        end    
   end
end   

no_match_images_AR=uint8(no_match_images_AR);
no_match_labels_AR=uint8(no_match_labels_AR);
a1=size(match_images_AR,3);
a2=size(no_match_images_AR,3);
train_x(:,:,1:a1,:)=match_images_AR;
train_x(:,:,a1+1:a1+a2,:)=no_match_images_AR;
train_y(1:a1)=match_labels_AR;
train_y(a1+1:a1+a2)=no_match_labels_AR;
clear match_images_AR;
clear no_match_images_AR;
clear match_labels_AR;
clear no_match_labels_AR;

%% AR测试数据集
load AR_face_image_test;

N=size(face_image,3);
rows=size(face_image,1);
cols=size(face_image,2);
M=14;%AR数据集每个人14幅图像
per_AR=8;%每个人8幅不同的图像对，组成不匹配的图像集合

% 获得匹配的图像对，
match_images_AR=zeros(rows,cols,M*(M-1)*(N/M)/2,2);
match_labels_AR=zeros(1,M*(M-1)*(N/M)/2);
for i=1:1:N/M %对于每个人
    temp=0;
    for j=1:1:M-1 %对于一个人的M幅图像
        for k=j+1:1:M
            temp=temp+1;
            match_images_AR(:,:,(i-1)*(M*(M-1)/2)+temp,1)=face_image(:,:,(i-1)*M+j);
            match_images_AR(:,:,(i-1)*(M*(M-1)/2)+temp,2)=face_image(:,:,(i-1)*M+k);
            match_labels_AR(1,(i-1)*(M*(M-1)/2)+temp)=1;%1表示是匹配图像
        end
    end
end
match_images_AR=uint8(match_images_AR);
match_labels_AR=uint8(match_labels_AR);

% 获得不是正确匹配的图像对
no_match_images_AR=zeros(rows,cols,(N/M)*M*per_AR,2);
no_match_labels_AR=zeros(1,(N/M)*M*per_AR);
for i=1:1:N/M %对于每个人,ORL是40个人
    for j=1:1:M %对于一个人的M幅图像，M是20
        %产生从1-100之间的10个不等于j的互不相等的数字
        while(1)
            rands=ceil(rand(1,per_AR)*20);
            a=find(rands==i);
            if(size(a,2)==0);
                break;
            end
        end
        for k=1:1:per_AR
            rand_a=ceil(rand(1)*M);%产生1-M之间的随机数
            no_match_images_AR(:,:,(i-1)*M*per_AR+(j-1)*per_AR+k,1)=face_image(:,:,(i-1)*M+j);
            no_match_images_AR(:,:,(i-1)*M*per_AR+(j-1)*per_AR+k,2)=face_image(:,:,(rands(k)-1)*M+rand_a);
            no_match_labels_AR(1,(i-1)*M*per_AR+(j-1)*per_AR+k)=0;
        end    
   end
end   

no_match_images_AR=uint8(no_match_images_AR);
no_match_labels_AR=uint8(no_match_labels_AR);
a1=size(match_images_AR,3);
a2=size(no_match_images_AR,3);
test_x(:,:,1:a1,:)=match_images_AR;
test_x(:,:,a1+1:a1+a2,:)=no_match_images_AR;
test_y(1:a1)=match_labels_AR;
test_y(a1+1:a1+a2)=no_match_labels_AR;

%% 保存训练和测试数据
save('AR_face_data_train.mat','train_x','train_y');
save('AR_face_data_test.mat','test_x','test_y');







