disp('### Beginning GPU unit tests ###');
dbstop if error

%dims = { [18 18 1], [128 99 1], [555 321 1] };
dims= { [12 12 12], [128 124 119], [333 69 100] };

%disp('-> Testing cudaApplyScalarPotential');

disp('-> Testing cudaArrayAtomic'); fail = 0;
for n = 1:numel(dims)
  res = dims{n};
  fprintf('  -> Resolution: [%i %i %i]\n', res(1), res(2), res(3));

  X = rand(res);
  Xg = GPU_Type(X);

  cudaArrayAtomic(Xg.GPU_MemPtr, .4, 1);
  X(X < .4) = .4;
  Xg.array(1);
  if any(any(any(Xg.array ~= X))); disp('  !!! Test failed: setting min !!!'); fail = 1; end

  cudaArrayAtomic(Xg.GPU_MemPtr, .6, 2);
  X(X > .6) = .6;
  Xg.array(1);
  if any(any(any(Xg.array ~= X))); disp('  !!! Test failed: setting max !!!'); fail = 1; end
  
  X( (X > .4) & (X < .6) ) = NaN;

  Xg.array = X;

  X(isnan(X)) = 0;
  cudaArrayAtomic(Xg.GPU_MemPtr, 0, 3);
  Xg.array(1);
  if any(any(any(Xg.array ~= X))); disp('  !!! Test failed: killing NaN !!!'); fail = 1; end
end

clear X; clear Xg;
if fail; disp('UNACCEPTABLE: cudaArrayAtomic failed'); return; end

disp('-> Testing cudaArrayRotate');
for n = 1:numel(dims)
  res = dims{n};
  fprintf('  -> Resolution: [%i %i %i]\n', res(1), res(2), res(3));

  X = rand(res);
  Xg = GPU_Type(X);
  Yg = GPU_Type(cudaArrayRotate(Xg.GPU_MemPtr,2));
  Yg.array(1);
  Xp = []; for z = 1:res(3); Xp(:,:,z) = transpose(X(:,:,z)); end
  if any(any(any(Yg.array ~= Xp)));  disp('  !!! Test failed: XY transpose !!!'); fail = 1; end
  clear Yg;

  if res(3) > 1
    Yg = GPU_Type(cudaArrayRotate(Xg.GPU_MemPtr,3));
    Yg.array(1);
    Xp = []; for z = 1:res(2); Xp(:,z,:) = squeeze(X(:,z,:))'; end
    if any(any(any(Yg.array ~= Xp))); disp('   !!! Test failed: XZ transpose !!!'); fail = 1; end
  end

end

clear X; clear Xg; clear Yg;
if fail; disp('UNACCEPTABLE: cudaArrayRotate failed'); return; end

disp('->  Testing freezeAndPtot');
for n = 1:numel(dims)
  res = dims{n};
  fprintf('  -> Resolution: [%i %i %i]\n', res(1), res(2), res(3));

  [xpos ypos zpos] = ndgrid((1:res(1))*2*pi/res(1), (1:res(2))*2*pi/res(2), (1:res(3))*2*pi/res(3));
  rho = ones(res);
  px = zeros(res) + sin(xpos);
  py = ones(res) + sin(ypos + zpos);
  pz = cos(zpos);
  E  = .5*(px.^2+py.^2+pz.^2)./rho + 2;
  Bx = zeros(res);
  By = zeros(res);
  Bz = zeros(res);

  disp('    Hydro');

  ptot = (2/3)*(E - .5*(px.^2+py.^2+pz.^2)./rho);
  if res(3) == 1
    freeze = max(sqrt((5/3)*ptot./rho) + abs(px./rho),[], 1)';
  else
    freeze = max(sqrt((5/3)*ptot./rho) + abs(px./rho),[], 1);
  end

  rhoD = GPU_Type(rho);
  pxD = GPU_Type(px); pyD = GPU_Type(py); pzD = GPU_Type(pz);
  ED = GPU_Type(E);
  BxD = GPU_Type(Bx); ByD = GPU_Type(By); BzD = GPU_Type(Bz);

  [pdev cdev] = freezeAndPtot(rhoD.GPU_MemPtr, ED.GPU_MemPtr, pxD.GPU_MemPtr, pyD.GPU_MemPtr, pzD.GPU_MemPtr, BxD.GPU_MemPtr, ByD.GPU_MemPtr, BzD.GPU_MemPtr, 5/3, 1);

  pd = GPU_Type(pdev);
  cf = GPU_Type(cdev);
  [ptot(1) pd.array(1) cf.array(1)]
  if max(max(max(abs(pd.array - ptot)))) > 1e-10; disp('   !!! Test failed: P !!!'); fail = 1; end
  if max(max(abs(squeeze(cf.array) - squeeze(freeze)))) > 1e-10; disp('   !!! Test failed: C_f !!!'); fail = 1; end

end

clear('rhoD','pxD','ED','pyD','pzD','BxD','ByD','BzD','pd','cf');
if fail; disp('UNACCEPTABLE: freeze speed failed'); return; end

disp('->  Testing cudaWstep:');
for n = 1:numel(dims)
  res = dims{n};
  fprintf('  -> Resolution: [%i %i %i]\n', res(1), res(2), res(3));

  [xpos ypos zpos] = ndgrid((1:res(1))*2*pi/res(1), (1:res(2))*2*pi/res(2), (1:res(3))*2*pi/res(3));
  rho = ones(res);
  px = zeros(res) + sin(xpos);
  py = ones(res) + sin(ypos + zpos);
  pz = cos(zpos);
  E  = .5*(px.^2+py.^2+pz.^2)./rho + 2;
  Bx = zeros(res);
  By = zeros(res);
  Bz = zeros(res);

  disp('    Hydro');

  ptot = (2/3)*(E - .5*(px.^2+py.^2+pz.^2)./rho);
  if res(3) == 1
    freeze = max(sqrt((5/3)*ptot./rho) + abs(px./rho),[], 1)';
  else
    freeze = max(sqrt((5/3)*ptot./rho) + abs(px./rho),[], 1);
  end

  rhoD = GPU_Type(rho);
  pxD = GPU_Type(px); pyD = GPU_Type(py); pzD = GPU_Type(pz);
  ED = GPU_Type(E);
  BxD = GPU_Type(Bx); ByD = GPU_Type(By); BzD = GPU_Type(Bz);
  PtotD = GPU_Type(ptot);
  freezespeed = GPU_Type(10*ones([size(rho,2) size(rho,3)]));

[rhow Ew pxw pyw pzw] = cudaFluidW(rhoD.GPU_MemPtr, ED.GPU_MemPtr, pxD.GPU_MemPtr, pyD.GPU_MemPtr, pzD.GPU_MemPtr, BxD.GPU_MemPtr, ByD.GPU_MemPtr, BzD.GPU_MemPtr, PtotD.GPU_MemPtr, freezespeed.GPU_MemPtr, .1, 1);

  % test stuff here
end

clear('rhoD','pxD','ED','pyD','pzD','BxD','ByD','BzD','PtotD','freezespeed');
if fail; disp('UNACCEPTABLE: cuda W step failed'); return; end




