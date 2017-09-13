function [er,bad,diff,out,br,FPR,TPR]=cnntest(net,x,y)
%该函数测试训练生成的卷积神经网络
%net表示训练完成的网络
%x表示输入测试图像，x=28x28x4060矩阵
%y表示测试输入x的目标输出,y=1x4060矩阵
%er表示错误率
%bad表示错误对的索引
%diff表示所有测试集合之间的欧式距离,[1x4060]
%out表示测试对的实际输出结果
%br是一个[1x2]向量，br(2)表示错误匹配对成为判为正确匹配对个数，
%br(1)表示正确匹配对判为错误匹配对个数
%FPR=(实际上不是同一个人，但是结果y=1的样本数目)/测试样本中负样本的数目
%TPR=(实际上是同一个人，且y=1的样本数目)/正样本数目

out=zeros(1,size(y,2));
net=cnnff(net,x);%计算输入x的输出
%net.o是10x10000矩阵，max计算列中最大值，返回行向量h,h(i)表示net.o的第i列最大值
%的索引
o1=net.o{1,1};
pow_o1=sqrt(sum(o1.*o1,1));
pow_o1=repmat(pow_o1,size(o1,1),1);
o1=o1./pow_o1;%归一化

o2=net.o{2,1};
pow_o2=sqrt(sum(o2.*o2,1));
pow_o2=repmat(pow_o2,size(o1,1),1);
o2=o2./pow_o2;

diff=sqrt(sum((o1-o2).^2,1));
[idx1]=find(diff<net.th);%输出为1的匹配对
out(idx1)=1;
[idx2]=find(diff>=net.th);%输出为0的匹配对
out(idx2)=0;

[idx_right_to_right]=find(diff<net.th & y==1);%实际上是正确匹配，输出为1的匹配对
[idx_right_to_wrong]=find(diff>=net.th & y==1);%实际上是正确匹配，输出是0的匹配对
[idx_wrong_to_right]=find(diff<net.th & y==0);%实际上是错误匹配对，输出是1的匹配对

bad1=size(idx_right_to_wrong,2);
bad2=size(idx_wrong_to_right,2);
right1=size(idx_right_to_right,2);

er=(bad1+bad2)/size(y,2);
bad=[idx_right_to_wrong,idx_wrong_to_right];
br=[bad1,bad2];
FPR=bad2/size(find(y==0),2);
TPR=right1/size(find(y==1),2);

%保存每对测试样本的距离
N=size(y,2);
fid = fopen('.\save_image\梦想的路_Distance.txt','wt');
fprintf(fid,'%.6f\n',net.th);%第一行是阈值
for k=1:N;
fprintf(fid,'%.6f\n',diff(k));
end
fclose(fid);

end


