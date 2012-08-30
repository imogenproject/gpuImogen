function propmon()
% We want to grab all 5 savefiles in the current directory, 

!grep waveK runfile.m | awk  '/.*/ { printf("%s %s ",$3,$4); }' | sed -e 's/\[//' > eviltmpfile;
!grep Mach runfile.m | awk  '/.*/ { printf("%s\n",$3); }' | sed -e 's/;//' >> eviltmpfile;
kandmach=load('eviltmpfile');
!rm -f eviltmpfile;

fnames = dir('2D_XY*');

amps = [];
times = [];

for n = 1:size(fnames,1)
    tempname = load(fnames(n).name);
    nom_de_plume = fieldnames(tempname);
    dataframe = getfield(tempname,nom_de_plume{1});

    amps(:,n) = wavetrack(dataframe, kandmach(1), kandmach(2), 5);
    times(n) = sum(dataframe.time.history);
end

[times key] = sort(times);

amps = amps(:,key);

%dt = diff(times);

da = diff(log(abs(amps)),1,2);
phase = unwrap(angle(amps),[],2);
%phase(phase < 0) = phase(phase < 0) + pi

dp = diff(phase,1,2);

%freq = sqrt(5/3)*sqrt(kandmach(1)^2+kandmach(2)^2); % The /2pi is killed by the 2pi^2 created in the |k| because have mode #s not wavevectors
damp = da(1,:);
dphi = dp(1,:);

% ./ dt(1,2:end);

    load('~/fastma_test.mat');
    stat_test.damp( (stat_test.kx==kandmach(1)), (stat_test.ky==kandmach(2)) ) = mean(damp(end)) / 2;%freq;
    stat_test.dphi( (stat_test.kx==kandmach(1)), (stat_test.ky==kandmach(2)) ) = mean(dphi(1:(end-1))) / 2;
    save('~/fastma_test.mat','stat_test');


end
