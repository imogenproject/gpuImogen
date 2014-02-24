%This programe is made by Nasar Ahmad, which creates Ascii Images
%Comsats Institue of Information Technology Islamabad (Telecom Eng III)
% Modified by Erik Keever

function ttplot(M)

% Get size of window
[~, tsys]= system('stty size');
termdim = sscanf(tsys, '%i %i');

% Affine transform M to lie within [0, 27]
M=M-min(M(:));
M=27*M/max(M(:));
gmax = 27;                  % Maximum number of intensity chracter

termdim=termdim'-2;
% If matrix is too large, rescale
if min(termdim ./ size(M)) < 1;
	image = round(imresize(M, min(termdim./size(M))));
else; image = round(M); end

[rows cols c]=size(image);
picture = char(zeros(rows,cols)); % creting a matrix where we can store characters   
for r = 1: rows
  for c = 1:cols
     B=image(r,c);%reading each pixel
     switch(B)
case(28)
A   =' ';
picture(r,c)=A;
case(27)
A	='.';
picture(r,c)=A;
case(26)
A	='`';
picture(r,c)=A;
case(25)
A	=char(39);
picture(r,c)=A;
case(24)
A	=':';
picture(r,c)=A;
case(23)
A	=',-^';       % all these characters are ' intensity wise '
picture(r,c)=A(round((rand(1)*2+1)));
case(22)                    
A	='!"~';            % This intensity measure are taking through a vb6 programe
picture(r,c)=A(round((rand(1)*2+1)));
case(21)
A	=';_';
picture(r,c)=A(round((rand(1)+1)));
case(20)
A	='()/<>\|';
picture(r,c)=A(round((rand(1)*6+1)));
case(19)
A	='*?';
picture(r,c)=A(round((rand(1)+1)));
case(18)
A	='=[]{}';
picture(r,c)=A(round((rand(1)*4+1)));
case(17)
A	='+';
picture(r,c)=A;
case(16)
A	='%17cilor';
picture(r,c)=A(round((rand(1)*7+1)));
case(15)
A	='$&Jjstx';
picture(r,c)=A(round((rand(1)*6+1)));
case(14)
A	='23Cfv';
picture(r,c)=A(round((rand(1)*4+1)));
case(13)
A	='5ILTeuz';
picture(r,c)=A(round((rand(1)*6+1)));
case(12)
A	='0469OYw'; % all these have same intensity ,so i am picking it randomly sothat
picture(r,c)=A(round((rand(1)*6+1)));%nobody thinks that this is wriiten by specific characters
case(11)
A	='8GPSXZany';
picture(r,c)=A(round((rand(1)*8+1)));
case(10)
A	='ADFVbd';
picture(r,c)=A(round((rand(1)*5+1)));
case(9)
A	='QRUghkpq';
picture(r,c)=A(round((rand(1)*7+1)));
case(8)
A	='m';
picture(r,c)=A;
case(7)
A	='E';
picture(r,c)=A;
case(6)
A	='BKW';
picture(r,c)=A(round((rand(1)*2+1)));
case(5)
A	='#';
picture(r,c)=A;
case(4)
A	='@';
picture(r,c)=A;
case(3)
A	='H';
picture(r,c)=A;
case(2)
A	='N';
picture(r,c)=A;
case(1)
A	='M';
picture(r,c)=A;
case(0)
A = ' ';
picture(r,c) = A;
end
  end
end

fprintf('\x1b[40m'); % set black BG
for i = 1:size(picture,1);
  outstring = '';
   for j = 1:size(picture,2);
    tocolor = floor(image(i,j)*6.99/27);
    switch(tocolor);
        case 0; outstring = [outstring sprintf('\x1b[31m%c', picture(i,j))];
	case 1; outstring = [outstring sprintf('\x1b[33m%c', picture(i,j))];
	case 2; outstring = [outstring sprintf('\x1b[32m%c', picture(i,j))];
	case 3; outstring = [outstring sprintf('\x1b[36m%c', picture(i,j))];
	case 4; outstring = [outstring sprintf('\x1b[34m%c', picture(i,j))];
	case 5; outstring = [outstring sprintf('\x1b[35m%c', picture(i,j))];
	case 6; outstring = [outstring sprintf('\x1b[37m%c', picture(i,j))];
    end
end; fprintf('%s\n',outstring); end

fprintf('\x1b[39m\x1b[49m');

end
