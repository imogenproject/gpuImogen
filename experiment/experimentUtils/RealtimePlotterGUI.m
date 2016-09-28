function fig = RealtimePlotterGUI(RTP)
% This function accepts Imogen's RealtimePlotter peripheral class as an argument,
% and throws up a pretty figure that lets you mess around with the realtime
% visualization subsystem as it runs (i.e. muck with the RTP's settings in realtime)
% This is, I am told, a useful thing in a program.

fig = findobj('Tag','ImogenRTP_GUIWindow');
if ~isempty(fig);
    fprintf('Realtime visualization control window already open. Returning...\n');
    return;
% FIXME: note that some of the stuff below oughta probably be run anyway in this case
end

fig = figure(32);
% turn off plot tools and resizer & give it a unique tag
fig.MenuBar = 'none';
fig.Resize = 'off';
fig.Name   = 'Realtime Plotter GUI';
fig.Tag    = 'ImogenRTP_GUIWindow';
p0 = fig.Position;
fig.Position = [p0(1) p0(2) 480 320];

worldPanel = uipanel(fig,'Title','Sim control', 'units', 'pixels', 'position', [255 175 215 140]);

% Button to control pause-on-call
button = uicontrol(worldPanel, 'Style','togglebutton','String','x','Position',[15 15 100 20], 'callback', @RTP.gcbSetPause);
if RTP.insertPause
    button.Value      = 1;
    button.BackgroundColor = [.3 .9 .3];
    button.String     = 'Pause-on-call';
else
    button.Value      = 0;
    button.BackgroundColor = [94 94 94]/100;
    button.String     = 'no pause-on-call';
end

button = uicontrol(worldPanel, 'Style','pushbutton','String','RESUME','tag','resumebutton','position',[120 15 80 20], 'callback', @RTP.gcbResumeFromSpin);

% Button controls force redraw on call
button = uicontrol(worldPanel, 'Style','togglebutton','String','x','Position',[15 40 185 20], 'callback', @RTP.gcbSetRedraw);
if RTP.insertPause
    button.Value      = 1; 
    button.BackgroundColor = [.3 .9 .3];
    button.String     = 'Force redraw';
else
    button.Value      = 0;
    button.BackgroundColor = [94 94 94]/100;
    button.String     = 'No forced redraw';
end

% Edit box for setting iterations/call
txt  = uicontrol(worldPanel, 'Style','text','String','Iterations/call:', 'position',[15 60 120 20]);
tbox = uicontrol(worldPanel, 'Style','edit','min',0,'max',1,'value',RTP.iterationsPerCall,'position',[130 65 70 20], 'callback',@RTP.gcbSetItersPerCall);

% Click to dump current RTP configuration to text
button = uicontrol(worldPanel,'Style','pushbutton','String','print config in console','position',[15 90 185 20], 'callback',@RTP.gcbDumpConfiguration);

%=========
plotPanel = uipanel(fig,'Title','Plot control', 'units', 'pixels', 'position', [15 15 225 300]);
    % control to pick how many plots (1, 2H, 2V, 4)
    button = uicontrol(plotPanel, 'Style','pushbutton','String','Cycle # of plots','Position',[15 230 195 20], 'callback', @RTP.gcbCycleNumOfPlots);

    % control to select which plot props to edit (1-4)
    button = uicontrol(plotPanel,'Style','pushbutton','String','Editing properties of plot 1','Position',[15 205 195 20], 'callback', @RTP.gcbCyclePlotSelection);

    % control to set plot's source fluid
    % FIXME implement callbacks for these in RTP...
    button = uicontrol(plotPanel,'Style','pushbutton','String','--','value',-1,'position',[15 180 20 20],'callback',@RTP.gcbSetPlotFluidsrc);
    txt    = uicontrol(plotPanel,'Style','text',      'String','Fluid: N','Tag','fluidnumbertxt','position',[40 175 60 20]);
    button = uicontrol(plotPanel,'Style','pushbutton','String','++','value',1,'position',[105 180 20 20],'callback',@RTP.gcbSetPlotFluidsrc);

    % control to set plot's qty to plot
    % FIXME identify what this passes in
    % FIXME Relabel vector quantities depending on geometry
    lis = uicontrol(plotPanel, 'Style','listbox',     'String','rho|px|py|pz|vx|vy|vz|Etotal|pressure|temperature','min',0,'max',1,'tag','qtylistbox','position', [15 15 100 150], 'callback', @RTP.gcbChoosePlotQuantity);

% These are arrayed right of the list and fluid # selector
    % flip 2d plots between imagesc and surf
    button = uicontrol(plotPanel, 'Style','pushbutton','String','imagesc','tag','plottypebutton','position',[130 90 80 20], 'callback', @RTP.gcbCyclePlotmode);

    % Toggle colorbar scale or not (grey out if 1D plot)
    button = uicontrol(plotPanel, 'Style','togglebutton','String','colorbar','tag','colorbarbutton','position',[130 65 80 20], 'callback', @RTP.gcbToggleColorbar);
    % Toggle drawing of grid or not
    button = uicontrol(plotPanel, 'Style','togglebutton','String','grid','tag','gridbutton','position',[130 40 80 20], 'callback', @RTP.gcbToggleGrid);
    % control to plot/image in log scale
    button = uicontrol(plotPanel, 'Style','togglebutton','String','log scale','tag','logbutton','position',[130 15 80 20], 'callback', @RTP.gcbToggleLogScale);

slicePanel = uipanel(fig,'title','Slicing control','units','pixels','position',[255 15 215 150]);
    btnX = uicontrol(slicePanel,'Style', 'togglebutton', 'String','X','tag','xSliceButton','position',[15 115 25 20],'callback',@RTP.gcbSetSlice);
    btnY = uicontrol(slicePanel,'Style', 'togglebutton', 'String','Y','tag','ySliceButton','position',[45 115 25 20],'callback',@RTP.gcbSetSlice);
    btnZ = uicontrol(slicePanel,'Style', 'togglebutton', 'String','Z','tag','zSliceButton','position',[75 115 25 20],'callback',@RTP.gcbSetSlice);
    btnXY = uicontrol(slicePanel,'Style','togglebutton', 'String','XY','tag','xySliceButton','position',[105 115 25 20],'callback',@RTP.gcbSetSlice);
    btnXZ = uicontrol(slicePanel,'Style','togglebutton', 'String','XZ','tag','xzSliceButton','position',[135 115 25 20],'callback',@RTP.gcbSetSlice);
    btnYZ = uicontrol(slicePanel,'Style','togglebutton', 'String','YZ','tag','yzSliceButton','position',[165 115 25 20],'callback',@RTP.gcbSetSlice);

    txt = uicontrol(slicePanel,'Style','text',           'String','Cut@','position',[35 90 50 20]);
    txt = uicontrol(slicePanel,'Style','text',           'String','| Subset A:B:C |','position',[90 90 100 20]);

    txt = uicontrol(slicePanel,'Style','text',           'String','X','position',[15 65 20 20]);
    txt = uicontrol(slicePanel,'Style','text',           'String','Y','position',[15 40 20 20]);
    txt = uicontrol(slicePanel,'Style','text',           'String','Z','position',[15 15 20 20]);

    % FIXME: these need to plug the existing default values into the text windows on startup... bleh.
    % NOTE: this should be run regardless of whether the window is new or not
    % 3x4 matrix of text areas: [cut   a:b:c] entries for each dimension
    for yn = 1:3;
        for xn = 4:-1:1;
            
            val = xn + 10*yn;
            tag = ['editcut' num2str(val)];
            area = uicontrol(slicePanel,'Style','edit','String','#','value',val,'tag',tag,'position',[(40*xn + 5*(xn>1)) (25*yn-10) 30 20],'callback',@RTP.gcbSetCuts);
            if xn == 1; area.String = num2str(RTP.cut(yn)); end
        end
    end


end
