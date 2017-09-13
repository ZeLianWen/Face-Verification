function  Y=rot180(X)
%该函数把X旋转180度
%Y是旋转后的输出结果

Y=flipdim(X,1);%上下翻转
Y=flipdim(Y,2);%左右翻转
end

