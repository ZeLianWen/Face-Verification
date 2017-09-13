function [y]=sigm(x)
%该函数根据输入值计算sigmod函数的输出值
%x表示函数输入
%y表示函数输出
y=1./(1+exp(-x));
end

