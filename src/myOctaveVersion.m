function result=myOctaveVersion()
%�ú�������OCTAVE_VERSION,����δ������ַ���

if isOctave()
    result=OCTAVE_VERSION;
else
    result='undefined';
end


