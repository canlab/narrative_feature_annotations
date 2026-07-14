function ax = plotFeatureMatrix(C, opts)
%PLOTFEATUREMATRIX Visualize the full feature time series, color-coded by category.
%   PLOTFEATUREMATRIX(C) draws a heatmap of the expanded feature matrix from
%   READANNOTATIONCORPUSFULL: time on the x-axis, one row per feature variable,
%   z-scored, with the variables grouped and a color strip on the left encoding the
%   six feature classes (visual/audio/language/social/situation/affect). This shows
%   all ~2.7k variables at once, organized by category.
%
%   PLOTFEATUREMATRIX(C, "Mode","classmean") instead plots, for each class, the mean
%   z-scored trajectory over time (six colored lines) — a compact category summary.
%
%   Name-value options:
%       "Clip"              ("")   restrict to one stimulus id (default: all clips)
%       "IncludeEmbeddings" (true) include SigLIP/DINOv2/CLAP dims in the heatmap
%       "Mode"              ("heatmap" | "classmean")
%       "SortBy"            ("classlevel" | "class")  row ordering within the heatmap
%       "CLim"              ([-3 3]) color limits for z-scored values
%       "SavePng"           ("")   if set, export the figure to this path
%
%   Returns the main axes handle.
%
%   Example:
%       C = readAnnotationCorpusFull("annotations/corpus");
%       plotFeatureMatrix(C, "Clip","BigBuckBunny");
%
%   See also READANNOTATIONCORPUSFULL, FEATUREINFO, FACTORANALYSISCORPUS.

arguments
    C (1,1) struct
    opts.Clip (1,1) string = ""
    opts.IncludeEmbeddings (1,1) logical = true
    opts.Mode (1,1) string {mustBeMember(opts.Mode, ["heatmap","classmean"])} = "heatmap"
    opts.SortBy (1,1) string {mustBeMember(opts.SortBy, ["classlevel","class"])} = "classlevel"
    opts.CLim (1,2) double = [-3 3]
    opts.SavePng (1,1) string = ""
end

% ---- select rows (timepoints) and columns (variables) ----
info = C.info;
cols = true(width(C.X), 1);
if ~opts.IncludeEmbeddings, cols = ~info.IsEmbedding; end
X = C.X(:, cols);
info = info(cols, :);

if opts.Clip ~= ""
    r = C.stim == opts.Clip;
    if ~any(r), error("plotFeatureMatrix:clip", "No clip '%s' in corpus.", opts.Clip); end
    X = X(r, :); t = C.time_sec(r); ttl = opts.Clip;
else
    t = (1:size(X,1))'; ttl = sprintf("%d clips concatenated", numel(C.ids));
end

% z-score each variable over the shown timepoints (ignore NaN)
Z = (X - mean(X, 1, "omitnan")) ./ std(X, 0, 1, "omitnan");
Z(:, ~isfinite(std(X,0,1,"omitnan"))) = 0;

% order variables by class (then level) so categories form contiguous bands
classes = ["visual","audio","language","social","situation","affect"];
levels  = ["low","mid","high"];
key = double(arrayfun(@(c) find(classes==c,1), info.Class));
if opts.SortBy == "classlevel"
    lvl = zeros(height(info),1);
    for i=1:height(info), lvl(i) = find(levels==info.Level(i),1); end
    [~, ord] = sortrows([key, lvl]);
else
    [~, ord] = sort(key);
end
Z = Z(:, ord); info = info(ord, :);

fig = figure("Color","w","Position",[100 100 1100 720]);
if opts.Mode == "classmean"
    ax = axes(fig); hold(ax,"on");
    for c = classes
        m = info.Class == c;
        if ~any(m), continue; end
        y = mean(Z(:, m), 2, "omitnan");
        rgb = info.Color(find(m,1), :);
        plot(ax, t, y, "Color", rgb, "LineWidth", 1.6, "DisplayName", c);
    end
    xlabel(ax, ternary(opts.Clip~="", "time (s)", "timepoint")); ylabel(ax, "mean z-score");
    legend(ax, "Location","eastoutside"); title(ax, "Class-mean feature trajectories — " + ttl);
    box(ax,"on");
else
    % main heatmap
    ax = axes(fig, "Position",[0.11 0.10 0.80 0.82]);
    imagesc(ax, "XData",[t(1) t(end)], "YData",[1 size(Z,2)], "CData", Z');
    set(ax, "YDir","reverse", "CLim", opts.CLim);
    colormap(ax, localDiverging());
    cb = colorbar(ax); cb.Label.String = "z-score";
    axis(ax, "tight");
    xlabel(ax, ternary(opts.Clip~="", "time (s)", "timepoint  (clips concatenated)"));
    ylabel(ax, "feature variables (grouped by category)");
    title(ax, sprintf("Feature matrix — %s  (%d variables x %d timepoints)", ttl, size(Z,2), size(Z,1)));

    % left color strip encoding class + class labels at band centers
    strip = axes(fig, "Position",[0.075 0.10 0.02 0.82]);
    cimg = reshape(info.Color, [height(info) 1 3]);
    image(strip, cimg); set(strip, "YDir","reverse");
    strip.XTick = []; strip.YTick = []; box(strip,"on");
    for c = classes
        m = find(info.Class == c);
        if isempty(m), continue; end
        yc = mean(m);
        text(strip, 0.5, yc, upper(extractBefore(c+" ",4)), "Rotation",90, ...
            "HorizontalAlignment","center","VerticalAlignment","middle", ...
            "FontWeight","bold","FontSize",8,"Color","w");
    end
    linkaxes([ax strip], "y"); ylim(strip, [0.5 height(info)+0.5]);
end

if opts.SavePng ~= ""
    exportgraphics(fig, opts.SavePng, "Resolution",150);
    fprintf("wrote %s\n", opts.SavePng);
end
end

% -------------------------------------------------------------------------
function m = localDiverging()
% blue-white-red diverging map without needing a toolbox
n = 256; x = linspace(0,1,n)';
r = min(1, 1.8*x); g = 1 - abs(2*x-1); b = min(1, 1.8*(1-x));
m = [r g b];
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
