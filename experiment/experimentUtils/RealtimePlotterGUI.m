function fig = RealtimePlotterGUI(RTP)
% This function accepts Imogen's RealtimePlotter peripheral class as an argument,
% and throws up a pretty figure that lets you mess around with the realtime
% visualization subsystem as it runs (i.e. muck with the RTP's settings in realtime)
% This is, I am told, a useful thing in a program.

fig = findobj('Tag','ImogenRTP_GUIWindow');
if ~isempty(fig)
    fprintf('Realtime visualization control window already open. Closing it.\n');
    close(fig);
end

core = groot;
DPI = core.ScreenPixelsPerInch;

eos = 10; % edge offset (from container e.g.)
txtH = round(.14*DPI); % assuming 10pt font
charW = round(.14*.6*DPI); % guess avg/char width
bh = round(11+txtH); % height of text boxes/buttons/etc
bs = 6; % box spacing
bsh=bs+bh;

halfColWidth = round(200*DPI/100);
fullColWidth = round(400*DPI/100 + eos);

winWidth = 2*halfColWidth + 3*eos;

col1_x0 = eos;
col2_x0 = 2*eos+halfColWidth; 

slicePanelHeight= 2*eos + 5*bsh+eos;
ctrlPanelHeight =  eos+4*bsh+eos;
plotPanelHeight = max(eos+10*bsh+2*eos, slicePanelHeight+eos+ctrlPanelHeight);
specPanelHeight = eos+bsh+bh+eos+bh;
moviePanelHeight = eos+bsh+bh+bh+eos;

winHeight = ctrlPanelHeight + plotPanelHeight + 3*eos + moviePanelHeight;

fig = figure(32);
% turn off plot tools and resizer & give it a unique tag
fig.MenuBar = 'none';
fig.Resize = 'off';
fig.Name   = 'Realtime Plotter GUI';
fig.Tag    = 'ImogenRTP_GUIWindow';
p0 = fig.Position;
fig.Position = [p0(1) p0(2) winWidth winHeight];

%=============================================================================================================
% Upper-right panel, controls 
worldPanel = uipanel(fig,'Title','Sim control', 'units', 'pixels', 'position', [col2_x0 (4*eos+specPanelHeight+slicePanelHeight+moviePanelHeight) halfColWidth ctrlPanelHeight]);

fw = halfColWidth - 2*eos;
hw = round((halfColWidth-2*eos-bs)/2);

% Button to control pause-on-call
button = uicontrol(worldPanel, 'Style','togglebutton','String','x','Position',[eos eos hw bh], 'callback', @RTP.gcbSetPause);
if RTP.insertPause
    button.Value      = 1;
    button.BackgroundColor = [.3 .9 .3];
    button.String     = 'Pause@call';
else
    button.Value      = 0;
    button.BackgroundColor = [94 94 94]/100;
    button.String     = 'No Pause@call';
end

button = uicontrol(worldPanel, 'Style','pushbutton','String','RESUME','tag','resumebutton','position',[eos+hw+bs, eos, hw, bh], 'callback', @RTP.gcbResumeFromSpin);

% Button controls force redraw on call
button = uicontrol(worldPanel, 'Style','togglebutton','String','x','Position',[eos (eos+bsh) (fw) bh], 'callback', @RTP.gcbSetRedraw);
if RTP.insertPause
    button.Value      = 1; 
    button.BackgroundColor = [.3 .9 .3];
    button.String     = 'Force redraw on call';
else
    button.Value      = 0;
    button.BackgroundColor = [94 94 94]/100;
    button.String     = 'No force redraw on call';
end

% Edit box for setting iterations/call
txt  = uicontrol(worldPanel, 'Style','text','String','Timesteps/call:', 'position',[eos (eos+2*bsh-round(bh/4)) (halfColWidth-2*eos-bs-6*charW) bh]);
tbox = uicontrol(worldPanel, 'Style','edit','min',0,'max',1,'value',RTP.iterationsPerCall,'string',num2str(RTP.iterationsPerCall), 'position',[halfColWidth-eos-6*charW, (eos+2*bsh), 6*charW, bh], 'callback',@RTP.gcbSetItersPerCall);

% Click to dump current RTP configuration to text
button = uicontrol(worldPanel,'Style','pushbutton','String','print cfg in term','position',[eos, (eos+3*bsh), halfColWidth-2*eos, bh], 'callback',@RTP.gcbDumpConfiguration);

%=======================================================================================================
% Left panel
plotPanel = uipanel(fig,'Title','Plot control', 'units', 'pixels', 'position', [col1_x0 (3*eos+specPanelHeight+moviePanelHeight) halfColWidth plotPanelHeight]);
    listHeight = 7*bsh-eos;
    fullw = halfColWidth - 2*eos;
    y0 = eos + listHeight;

    button = uicontrol(plotPanel, 'Style','pushbutton','String','Rearrange plots', 'Position',[eos (y0+eos+3*bsh) fullw bh], 'callback', @RTP.gcbCyclePlotOffset);
    % control to pick how many plots (1, 2H, 2V, 4)
    button = uicontrol(plotPanel, 'Style','pushbutton','String','Cycle # of plots','Position',[eos (y0+eos+2*bsh) fullw bh], 'callback', @RTP.gcbCycleNumOfPlots);

    % control to select which plot props to edit (1-4)
    button = uicontrol(plotPanel,'Style','pushbutton','String','Editing plot 1','Position',[eos (y0+eos+bsh) fullw bh], 'callback', @RTP.gcbCyclePlotSelection);

    % control to set plot's source fluid
    % FIXME implement callbacks for these in RTP...
    button = uicontrol(plotPanel,'Style','pushbutton','String','--','UserData',-1,'position',[eos (y0+eos) 5*charW bh],'callback',@RTP.gcbSetPlotFluidsrc);
    txt    = uicontrol(plotPanel,'Style','text',      'String','Fluid: N','Tag','fluidnumbertxt','position',[(eos+5*charW+bs) (y0+eos-5) (fullw-10*charW) bh]);
    button = uicontrol(plotPanel,'Style','pushbutton','String','++','UserData',1,'position',[(fullw+eos-5*charW) (y0+eos) 5*charW bh],'callback',@RTP.gcbSetPlotFluidsrc);

    % control to set plot's qty to plot
    % FIXME identify what this passes in
    % FIXME Relabel vector quantities depending on geometry
    tWidth = 12+8*charW;
    tost = halfColWidth - eos - tWidth;
    listWidth = halfColWidth - tWidth - 3*eos;
    lis = uicontrol(plotPanel, 'Style','listbox',     'String','rho|px|py|pz|vx|vy|vz|Etotal|Pressure|Temp|Stokes #|Drag time','min',0,'max',1,'tag','qtylistbox','position', [eos eos listWidth listHeight], 'callback', @RTP.gcbChoosePlotQuantity);

% These are arrayed right of the list and fluid # selector
    button = uicontrol(plotPanel, 'Style','togglebutton','String','equal axes','tag','eqaxesbutton','position',[tost (eos+6*bsh) tWidth bh], 'callback', @RTP.gcbToggleEqualAxes);
    button = uicontrol(plotPanel, 'Style','togglebutton','String','V field','tag','velfieldbutton','position',[tost (eos+5*bsh) tWidth bh], 'callback', @RTP.gcbToggleVelocityField);
    % Alternate axis labelling (none, pixels, position)
    button = uicontrol(plotPanel, 'Style','pushbutton','String','axis off','tag','axeslabelsbutton','position',[tost (eos+4*bsh) tWidth bh], 'callback', @RTP.gcbCycleAxisLabels);
    % flip 2d plots between imagesc and surf
    button = uicontrol(plotPanel, 'Style','pushbutton','String','imagesc','tag','plottypebutton','position',[tost (eos+3*bsh) tWidth bh], 'callback', @RTP.gcbCyclePlotmode);
    % Toggle colorbar scale or not (grey out if 1D plot)
    button = uicontrol(plotPanel, 'Style','togglebutton','String','colorbar','tag','colorbarbutton','position',[tost (eos+2*bsh) tWidth bh], 'callback', @RTP.gcbToggleColorbar);
    % Toggle drawing of grid or not
    button = uicontrol(plotPanel, 'Style','togglebutton','String','grid','tag','gridbutton','position',[tost (eos+bsh) tWidth bh], 'callback', @RTP.gcbToggleGrid);
    % control to plot/image in log scale
    button = uicontrol(plotPanel, 'Style','togglebutton','String','log scale','tag','logbutton','position',[tost eos tWidth bh], 'callback', @RTP.gcbToggleLogScale);

%===================================================================================================
% This is the slicing control panel, lower right...
slicePanel = uipanel(fig,'title','Slicing control','units','pixels','position',[col2_x0 (3*eos+specPanelHeight+moviePanelHeight) halfColWidth slicePanelHeight]);
    bw = 3*charW; % button width
    bws = bw + bs; % button width spacing

    btnX = uicontrol(slicePanel,'Style', 'togglebutton', 'String','X', 'tag','xSliceButton', 'position',[eos        , eos+4*bsh-bs, bw bh],'callback',@RTP.gcbSetSlice);
    btnY = uicontrol(slicePanel,'Style', 'togglebutton', 'String','Y', 'tag','ySliceButton', 'position',[(eos+1*bws), eos+4*bsh-bs, bw bh],'callback',@RTP.gcbSetSlice);
    btnZ = uicontrol(slicePanel,'Style', 'togglebutton', 'String','Z', 'tag','zSliceButton', 'position',[(eos+2*bws), eos+4*bsh-bs, bw bh],'callback',@RTP.gcbSetSlice);
    btnXY = uicontrol(slicePanel,'Style','togglebutton', 'String','XY','tag','xySliceButton','position',[(eos+3*bws), eos+4*bsh-bs, bw bh],'callback',@RTP.gcbSetSlice);
    btnXZ = uicontrol(slicePanel,'Style','togglebutton', 'String','XZ','tag','xzSliceButton','position',[(eos+4*bws), eos+4*bsh-bs, bw bh],'callback',@RTP.gcbSetSlice);
    btnYZ = uicontrol(slicePanel,'Style','togglebutton', 'String','YZ','tag','yzSliceButton','position',[(eos+5*bws), eos+4*bsh-bs, bw, bh],'callback',@RTP.gcbSetSlice);

    txt = uicontrol(slicePanel,'Style','text',           'String','Cut','position',[eos+charW+bs, eos+3*bsh-bs 4*charW bh]);
    txt = uicontrol(slicePanel,'Style','text',           'String','Subset, A:B:C','position',[eos+bs+6*charW, eos+3*bsh-bs, 12*charW, bh]);

    txt = uicontrol(slicePanel,'Style','text',           'String','X','position',[eos (eos+2*bsh-4) charW bh]);
    txt = uicontrol(slicePanel,'Style','text',           'String','Y','position',[eos (eos+bsh-4) charW bh]);
    txt = uicontrol(slicePanel,'Style','text',           'String','Z','position',[eos eos-4 charW bh]);

    % FIXME: these need to plug the existing default values into the text windows on startup... bleh.
    % NOTE: this should be run regardless of whether the window is new or not
    % 3x4 matrix of text areas: [cut   a:b:c] entries for each dimension
    for yn = 1:3
        for xn = 4:-1:1
            
            val = xn + 10*yn;
            tag = ['editcut' num2str(val)];
	    if xn >= 2
                inistr = num2str(RTP.indSubs(4-yn, xn-1));
	    else
		inistr = num2str(RTP.cut(4-yn));
	    end
            area = uicontrol(slicePanel,'Style','edit','String',inistr,'value',val,'tag',tag,'position',[(eos + charW + bs + (xn-1)*(4*charW+bs)), (bsh*(yn-1)+eos), 4*charW, bh],'callback',@RTP.gcbSetCuts);
        end
    end

specPanelVF = uipanel(fig,'title','Velocity field overlay control','units','pixels','position',[col1_x0 2*eos+moviePanelHeight fullColWidth specPanelHeight]);
    txt = uicontrol(specPanelVF, 'Style','text','String','Downsample', 'position',[eos eos 120 bh]);
    tbox= uicontrol(specPanelVF, 'Style','edit','min',1,'value',10,'string','10','tag','decfactorbox', 'position',[2*eos+120, eos, 30 bh], 'callback',@RTP.gcbSetVF_df);
    btn = uicontrol(specPanelVF, 'Style', 'pushbutton','String','COLOR','tag','vf_colorbutton','position',[eos, (2*eos+bh) 80 bh], 'callback', @RTP.gcbCycleVF_color);

    btn = uicontrol(specPanelVF, 'Style', 'pushbutton','String','Heavier','tag','vf_heavybutton','position',[eos+200 (eos) 80 bh], 'callback', @RTP.gcbSetVF_weight);
    btn = uicontrol(specPanelVF, 'Style', 'pushbutton','String','Finer','tag','vf_lightbutton','position',[eos+200 (2*eos+bh) 80 bh], 'callback', @RTP.gcbSetVF_weight);

    btn = uicontrol(specPanelVF, 'Style', 'pushbutton','String','Velocity','tag','vf_typebutton','position',[eos+300 (2*eos+bh) 100 bh], 'callback', @RTP.gcbCycleVF_type);

moviePanel = uipanel(fig, 'title','Movie frame output control','units','pixels','position',[eos eos fullColWidth, moviePanelHeight]);
    qw = round((fullColWidth - 2*eos - 2*bs)/4);
    hw = round((fullColWidth - 2*eos - bs)/2);

    btn = uicontrol(moviePanel, 'Style','togglebutton', 'String','Not writing imgs', 'tag', 'imgWriteButton','position',[eos, eos+bsh, hw, bh],'callback',@RTP.gcbMovieToggle);

    txt = uicontrol(moviePanel, 'Style','text','String','Prefix:','position',[eos, eos, qw, bh]);
    txt = uicontrol(moviePanel, 'Style','edit','String','RTP_','tag','movieprefixfield','position',[eos+qw+bs, eos, qw, bh], 'callback',@RTP.gcbMoviePrefix);

    btn = uicontrol(moviePanel, 'Style','text','String','Next frame: 0', 'tag','movieframereport','position',[eos+hw+bs, eos+bsh-txtH/2, hw, bh]);

    txt = uicontrol(moviePanel, 'Style','text','String','Set frame:','position',[eos+2*bs+2*qw, eos, qw, bh]);
    btn = uicontrol(moviePanel, 'Style','edit','String','asdf','tag','movieframesetter','position',[eos+2*bs+3*qw, eos, qw, bh],'callback',@RTP.gcbMovieSetFrame);

   %set frame (text input)

end
