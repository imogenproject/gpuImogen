% Function simply invokes the GPU code unit test suite
gm = GPUManager.getInstance();

try
    gm.pushParameters();
    tortureTest(gm.deviceList,'y',5);
    gm.popParameters();
catch oh_teh_noez
    prettyprintException(oh_teh_noez, 1, 'One or more ranks have failed GPU tests! ECC? Probably hardware problem. Otherwise -> Bad code checked into git, smack devs on nose with newspaper');
    rethrow(oh_teh_noez);
end


