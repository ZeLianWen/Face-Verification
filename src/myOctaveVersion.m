function result=myOctaveVersion()
%该函数返回OCTAVE_VERSION,或者未定义的字符串

if isOctave()
    result=OCTAVE_VERSION;
else
    result='undefined';
end


