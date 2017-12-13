function x = date2ymdhms(d)

if numel(d) ~= 6
    error('argument must be a 6-element vector; see ''doc clock''');
end

e = int32(d);

x = sprintf('%04i%02i%02i_%02i%02i%02i', e(1), e(2), e(3), e(4), e(5), e(6));

end