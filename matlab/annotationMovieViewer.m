function annotationMovieViewer(movieFile, annPath, opts)
%ANNOTATIONMOVIEVIEWER Play a movie with its annotation time series synced below.
%   ANNOTATIONMOVIEVIEWER(MOVIEFILE, ANNPATH) opens a window showing the movie on
%   top and a stack of annotation time series below, with a red vertical marker on
%   each series tracking the current playback position. Play/Pause and a scrub
%   slider let you move through the stimulus; the marker and video stay in sync.
%
%   ANNPATH may be an annotation .h5, its folder, or a JSON profile.
%
%   Name-value options:
%       "Channels"  string array of channel paths to display
%                   (default: a representative set of applicable scalar channels)
%       "Speed"     playback speed multiplier (default 1.0)
%
%   Example:
%       d = "annotations/corpus/ses-01_run-01_order-04_content-parkour";
%       m = "data/movies/spacetop/videos/ses-01/ses-01_run-01_order-04_content-parkour.mp4";
%       annotationMovieViewer(m, d)
%
%   See also READANNOTATIONS, READANNOTATIONCORPUS, GETFEATURE.

arguments
    movieFile (1,1) string {mustBeFile}
    annPath (1,1) string
    opts.Channels (1,:) string = string.empty
    opts.Speed (1,1) double = 1.0
end

ann = readAnnotations(annPath);
% Audio-only / text-only stimuli have no video stream: degrade to a
% time-series-only viewer (marker + scrub still work) instead of erroring.
vid = [];
try
    vid = VideoReader(char(movieFile));
catch
end
hasVid = ~isempty(vid);
[chans, series] = pickChannels(ann, opts.Channels);
nCh = numel(chans);
t_ann = ann.time_sec(:);
if hasVid
    tmax = min(vid.Duration, max(t_ann) + 1);
else
    tmax = max(t_ann) + 1;
end

% --- shared playback state (mutated by nested callbacks) -------------------
tNow = 0;

% --- layout ---------------------------------------------------------------
fig = figure("Name", "Annotation viewer: " + ann.stimulus.id, "Color", "w", ...
    "Position", [100 80 900 760], "CloseRequestFcn", @(~, ~) onClose());
tl = tiledlayout(fig, nCh + 5 * hasVid, 1, "TileSpacing", "tight", "Padding", "compact");

imh = [];
if hasVid
    axV = nexttile(tl, [5 1]);
    imh = imshow(readFrame(vid), "Parent", axV);
    vid.CurrentTime = 0;
    title(axV, ann.stimulus.id, "Interpreter", "none");
end

mk = gobjects(1, nCh);
axCh = gobjects(1, nCh);
for k = 1:nCh
    ax = nexttile(tl);
    if k == 1 && ~hasVid
        title(ax, ann.stimulus.id + "  (no video — time series only)", "Interpreter", "none");
    end
    plot(ax, t_ann, series{k}, "k");
    ax.XLim = [0 tmax];
    ax.YTick = [];
    ylabel(ax, shortName(chans(k)), "Interpreter", "none", "Rotation", 0, ...
        "HorizontalAlignment", "right", "VerticalAlignment", "middle");
    if k < nCh
        ax.XTickLabel = [];
    end
    hold(ax, "on");
    mk(k) = xline(ax, 0, "r", "LineWidth", 1.5);
    axCh(k) = ax;
end
xlabel(axCh(end), "time (s)");

% --- controls -------------------------------------------------------------
btn = uicontrol(fig, "Style", "togglebutton", "String", "Play", "Units", "normalized", ...
    "Position", [0.02 0.005 0.08 0.035], "Callback", @(s, ~) onPlay(s));
sld = uicontrol(fig, "Style", "slider", "Units", "normalized", "Min", 0, "Max", tmax, ...
    "Value", 0, "Position", [0.12 0.005 0.76 0.035], "Callback", @(s, ~) onSeek(s));
lbl = uicontrol(fig, "Style", "text", "Units", "normalized", "BackgroundColor", "w", ...
    "Position", [0.89 0.005 0.10 0.035], "String", "0.0 s");

% Create the timer only after all setup succeeded (an earlier error would leak it).
tmr = timer("ExecutionMode", "fixedRate", "Period", 0.05, "BusyMode", "drop", ...
    "TimerFcn", @(~, ~) onTick());

refresh();

    % ===================== nested callbacks ===========================
    function refresh()
        if ~isvalid(fig)                    % figure force-closed: stop + clean the timer
            try, stop(tmr); delete(tmr); catch, end %#ok<CTCH>
            return
        end
        tNow = max(0, min(tNow, tmax));
        if hasVid
            ct = min(tNow, vid.Duration - 1 / max(vid.FrameRate, 1));
            try
                vid.CurrentTime = max(ct, 0);
                if hasFrame(vid)
                    imh.CData = readFrame(vid);
                end
            catch
            end
        end
        for j = 1:nCh
            mk(j).Value = tNow;
        end
        sld.Value = min(tNow, tmax);
        lbl.String = sprintf("%.1f s", tNow);
        drawnow limitrate;
    end

    function onTick()
        if ~isvalid(fig)
            try, stop(tmr); delete(tmr); catch, end %#ok<CTCH>
            return
        end
        tNow = tNow + tmr.Period * opts.Speed;
        if tNow >= tmax
            tNow = tmax;
            stop(tmr);
            btn.Value = 0;
            btn.String = "Play";
        end
        refresh();
    end

    function onPlay(src)
        if src.Value
            src.String = "Pause";
            if tNow >= tmax
                tNow = 0;
            end
            start(tmr);
        else
            src.String = "Play";
            stop(tmr);
        end
    end

    function onSeek(src)
        tNow = src.Value;
        refresh();
    end

    function onClose()
        try
            stop(tmr);
        catch
        end
        delete(tmr);
        delete(fig);
    end
end

% ======================= local helpers ================================
function [chs, ser] = pickChannels(ann, requested)
preferred = ["visual/low_level_static/luminance", ...
    "visual/dynamic_motion/flow_magnitude", ...
    "audio/low_level/rms", ...
    "audio/speech/word_rate", ...
    "audio/speech/voice_valence", ...
    "affect/depicted/vlm_valence", ...
    "visual/faces_bodies_gaze/n_faces", ...
    "situation/event_boundary"];
if ~isempty(requested)
    preferred = requested;
end
n = numel(ann.time_sec);
chs = strings(1, 0);
ser = {};
for c = preferred
    try
        ch = getFeature(ann, c);
    catch
        continue
    end
    v = ch.value;
    if isnumeric(v) && isvector(v) && numel(v) == n ...
            && (~isfield(ch, "applicable") || ch.applicable) && ~all(isnan(double(v)))
        chs(end+1) = c; %#ok<AGROW>
        ser{end+1} = double(v(:)); %#ok<AGROW>
        if numel(chs) >= 6 && isempty(requested)
            break
        end
    end
end
if isempty(chs)
    error("annotationMovieViewer:noChannels", "No displayable scalar channels found.");
end
end

function s = shortName(path)
parts = split(string(path), "/");
if numel(parts) >= 2
    s = parts(end-1) + "/" + parts(end);
else
    s = parts(end);
end
end
