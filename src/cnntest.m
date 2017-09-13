function [er,bad,diff,out,br,FPR,TPR]=cnntest(net,x,y)
%�ú�������ѵ�����ɵľ��������
%net��ʾѵ����ɵ�����
%x��ʾ�������ͼ��x=28x28x4060����
%y��ʾ��������x��Ŀ�����,y=1x4060����
%er��ʾ������
%bad��ʾ����Ե�����
%diff��ʾ���в��Լ���֮���ŷʽ����,[1x4060]
%out��ʾ���ԶԵ�ʵ��������
%br��һ��[1x2]������br(2)��ʾ����ƥ��Գ�Ϊ��Ϊ��ȷƥ��Ը�����
%br(1)��ʾ��ȷƥ�����Ϊ����ƥ��Ը���
%FPR=(ʵ���ϲ���ͬһ���ˣ����ǽ��y=1��������Ŀ)/���������и���������Ŀ
%TPR=(ʵ������ͬһ���ˣ���y=1��������Ŀ)/��������Ŀ

out=zeros(1,size(y,2));
net=cnnff(net,x);%��������x�����
%net.o��10x10000����max�����������ֵ������������h,h(i)��ʾnet.o�ĵ�i�����ֵ
%������
o1=net.o{1,1};
pow_o1=sqrt(sum(o1.*o1,1));
pow_o1=repmat(pow_o1,size(o1,1),1);
o1=o1./pow_o1;%��һ��

o2=net.o{2,1};
pow_o2=sqrt(sum(o2.*o2,1));
pow_o2=repmat(pow_o2,size(o1,1),1);
o2=o2./pow_o2;

diff=sqrt(sum((o1-o2).^2,1));
[idx1]=find(diff<net.th);%���Ϊ1��ƥ���
out(idx1)=1;
[idx2]=find(diff>=net.th);%���Ϊ0��ƥ���
out(idx2)=0;

[idx_right_to_right]=find(diff<net.th & y==1);%ʵ��������ȷƥ�䣬���Ϊ1��ƥ���
[idx_right_to_wrong]=find(diff>=net.th & y==1);%ʵ��������ȷƥ�䣬�����0��ƥ���
[idx_wrong_to_right]=find(diff<net.th & y==0);%ʵ�����Ǵ���ƥ��ԣ������1��ƥ���

bad1=size(idx_right_to_wrong,2);
bad2=size(idx_wrong_to_right,2);
right1=size(idx_right_to_right,2);

er=(bad1+bad2)/size(y,2);
bad=[idx_right_to_wrong,idx_wrong_to_right];
br=[bad1,bad2];
FPR=bad2/size(find(y==0),2);
TPR=right1/size(find(y==1),2);

%����ÿ�Բ��������ľ���
N=size(y,2);
fid = fopen('.\save_image\�����·_Distance.txt','wt');
fprintf(fid,'%.6f\n',net.th);%��һ������ֵ
for k=1:N;
fprintf(fid,'%.6f\n',diff(k));
end
fclose(fid);

end


