%检测是否正在倍频运行
function result=isOctave()

result=exist('OCTAVE_VERSION')~=0;
end


