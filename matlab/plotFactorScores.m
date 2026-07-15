function F = plotFactorScores(opts)
%PLOTFACTORSCORES Load the per-category factor scores and visualize them 6 ways.
%   F = PLOTFACTORSCORES() loads analysis/extracted_factors.mat (written by
%   EXTRACTCATEGORYFACTORS) and produces six figures of the 129 per-model factors,
%   color-coded by feature category, using CANlab core visualization tools. It saves
%   an .svg and .png of each figure to matlab/figures/ and returns the factor-score
%   struct F (augmented with the category colors, the correlation matrix, and the
%   t-SNE / UMAP embeddings it computes).
%
%   Figures:
%     1. Time series of all factors for one clip, each factor a horizontal trace
%        (plot_matrix_cols), colored by category (10 same-colored lines per model).
%     2. Heatmap of the factor scores (z-scored per factor, since factoran/PCA scores
%        are on different scales), with CANlab's "mango" split colormap.
%     3. Force-directed graph of the factor-factor correlations (canlab_force_directed_graph),
%        nodes colored by category.
%     4. Correlation matrix (plot_correlation_matrix) with per-category color bars/labels.
%     5. t-SNE of the factors (each factor a point), colored by category.
%     6. UMAP of the factors (run_umap), colored by category.
%
%   Units: figures 3-6 treat each FACTOR as the unit (129 nodes/points) so "category"
%   is a meaningful color; figures 1-2 show the factor time series over timepoints.
%
%   Name-value options:
%       "MatFile"    ("analysis/extracted_factors.mat")
%       "OutDir"     ("matlab/figures")
%       "CanlabPath" ("/Users/f003vz1/Documents/GitHub/CanlabCore/CanlabCore")
%       "Clip"       ("BigBuckBunny")   audiovisual clip for the time-series figure
%       "Save"       (true)             write .svg + .png for each figure
%
%   Colors match the six feature domains in the feature summary table (visual, audio,
%   language, social, situation, affect), with lighter tints separating the categories
%   that share a domain.
%
%   Example:
%       F = plotFactorScores();
%
%   See also EXTRACTCATEGORYFACTORS, READANNOTATIONCORPUSFULL, PLOT_MATRIX_COLS,
%   CANLAB_FORCE_DIRECTED_GRAPH, PLOT_CORRELATION_MATRIX.

arguments
    opts.MatFile (1,1) string = "analysis/extracted_factors.mat"
    opts.OutDir (1,1) string = "matlab/figures"
    opts.CanlabPath (1,1) string = "/Users/f003vz1/Documents/GitHub/CanlabCore/CanlabCore"
    opts.Clip (1,1) string = "BigBuckBunny"
    opts.Save (1,1) logical = true
end

if isfolder(opts.CanlabPath), addpath(genpath(char(opts.CanlabPath))); end
if ~isfolder(opts.OutDir), mkdir(opts.OutDir); end

S = load(opts.MatFile);
F = S.extracted_factors;
Y = F.scores;                         % [T x 129]
[T, nF] = size(Y);

% ---- per-factor category + color (domain hue, lighter tints within a domain) --------
cats   = string({F.byCategory.name});
nf     = [F.byCategory.nFactors];
catCls = string({F.byCategory.class});
catIdx = repelem(1:numel(cats), nf)';           % [nF x 1] category per factor
catRGB = localCategoryColors(cats, catCls);     % [nCat x 3]
factorRGB = catRGB(catIdx, :);                  % [nF x 3]
F.category_colors = catRGB;
F.category_names  = cats;
F.factor_category = catIdx;

catColorsCell = arrayfun(@(i) catRGB(i,:), 1:numel(cats), "UniformOutput", false);
catLabels = cellstr(strrep(cats, "_", " "));

fprintf("plotFactorScores: %d factors in %d categories, %d timepoints.\n", nF, numel(cats), T);

% ===================================================================================
% 1. TIME SERIES for one clip: each factor a horizontal trace, colored by category
% ===================================================================================
r = F.stim == opts.Clip;
if ~any(r)
    warning("Clip '%s' not found; using the first clip.", opts.Clip);
    r = F.stim == F.stim(1); opts.Clip = string(F.stim(1));
end
Zc = zscore(Y(r, :));                            % z-score each factor within the clip
tsec = F.time_sec(r);
colorsCell = arrayfun(@(i) factorRGB(i,:), 1:nF, "UniformOutput", false);

f1 = figure("Color","w","Position",[80 80 1200 780]);
plot_matrix_cols(Zc, "horiz", tsec, colorsCell, 1);
set(gca, "YTick", []); xlabel("time (s)"); ylabel("factors (grouped by category)");
title(sprintf("Factor time series — %s   (%d factors, colored by category)", opts.Clip, nF));
localCategoryLegend(gca, cats, catRGB);
localSave(f1, opts, "01_factor_timeseries");

% ===================================================================================
% 2. HEATMAP of z-scored factor scores, mango colormap (scores are on different scales)
% ===================================================================================
Zall = zscore(Y);                                % normalize: factoran/PCA scores differ in scale
f2 = figure("Color","w","Position",[80 80 1180 760]);
ax = axes(f2, "Position",[0.10 0.10 0.82 0.82]);
imagesc(ax, [1 T], [1 nF], Zall.');
set(ax, "YDir","reverse", "CLim",[-3 3]);
colormap(ax, localMango(256)); cb = colorbar(ax); cb.Label.String = "z-scored factor score";
xlabel(ax, "timepoint  (all clips concatenated)"); ylabel(ax, "factor");
title(ax, sprintf("Factor scores heatmap  (%d factors x %d timepoints, mango colormap)", nF, T));
strip = axes(f2, "Position",[0.068 0.10 0.02 0.82]);
image(strip, reshape(factorRGB, [nF 1 3])); set(strip, "YDir","reverse");
strip.XTick = []; strip.YTick = []; box(strip,"on"); linkaxes([ax strip], "y"); ylim(strip,[0.5 nF+0.5]);
localSave(f2, opts, "02_factor_heatmap");

% ===================================================================================
% 3. FORCE-DIRECTED GRAPH of factor-factor correlations, colored by category
% ===================================================================================
% canlab_force_directed_graph creates its own figure; capture it with gcf.
stats = canlab_force_directed_graph(Y, 'partitions', catIdx, ...
    'partitioncolors', catColorsCell, 'names', cellstr(F.labels.FactorName), 'linewidth', 0.5);
f3 = gcf; set(f3, "Color","w", "Position",[80 80 900 820]);
title(sprintf("Force-directed graph of %d factors (colored by category)", nF));
F.graph_stats = stats;
localCategoryLegend(gca, cats, catRGB);
localSave(f3, opts, "03_force_directed_graph");

% ===================================================================================
% 4. CORRELATION MATRIX with per-category color bars + labels
% ===================================================================================
% plot_correlation_matrix makes its own figure ('plotmatrix'); capture it with gcf.
plot_correlation_matrix(full(double(stats.C)), 'input_is_r', true, 'docircles', false, 'doimage', true, ...
    'partitions', catIdx(:)', 'partitioncolors', catColorsCell, 'partitionlabels', catLabels);
f4 = gcf; set(f4, "Color","w", "Position",[80 80 980 900]);
title(sprintf("Factor correlation matrix  (%d factors, grouped by category)", nF));
localSave(f4, opts, "04_correlation_matrix");

% ===================================================================================
% 5. t-SNE of the factors (each factor a point), colored by category
% ===================================================================================
rng(42);
perplex = min(30, floor((nF-1)/3));
Ytsne = tsne(Y.', "Standardize", true, "Perplexity", perplex);
F.tsne = Ytsne;
f5 = localScatter2D(Ytsne, factorRGB, cats, catRGB, catIdx, ...
    sprintf("t-SNE of %d factors (perplexity %d)", nF, perplex));
localSave(f5, opts, "05_tsne");

% ===================================================================================
% 6. UMAP of the factors, colored by category
% ===================================================================================
try
    rng(42);
    close all;                                          % umap toolbox needs a clean figure state
    reduction = run_umap(Y.', 'method', 'Java', 'n_neighbors', min(15, nF-1), ...
        'min_dist', 0.3, 'verbose', 'none');            % (figures 1-5 already saved to disk)
    close all;                                          % drop any popup umap created
    F.umap = reduction;
    f6 = localScatter2D(reduction, factorRGB, cats, catRGB, catIdx, ...
        sprintf("UMAP of %d factors", nF));
    localSave(f6, opts, "06_umap");
catch ME
    warning("plotFactorScores:umap", "UMAP step skipped (%s).", ME.message);
end

fprintf("Done. Figures saved to %s (svg + png).\n", opts.OutDir);
end

% =====================================================================================
% helpers
% =====================================================================================
function RGB = localCategoryColors(cats, catCls)
% Domain hue per class (matches the feature summary table); lighter tints separate the
% categories that share a domain.
dom = containers.Map(...
    {'visual','audio','language','social','situation','affect','(mixed)'}, ...
    {[.39 .40 .95],[.02 .71 .83],[.96 .62 .04],[.93 .28 .60],[.06 .73 .51],[.94 .27 .27],[.39 .45 .55]});
RGB = zeros(numel(cats), 3);
uc = unique(catCls, "stable");
for c = uc
    idx = find(catCls == c);
    base = dom(char(c));
    f = linspace(1.0, 0.45, numel(idx));                 % 1 = full hue, <1 = lighter tint
    for j = 1:numel(idx)
        RGB(idx(j), :) = base*f(j) + (1 - f(j))*[1 1 1]; % blend toward white
    end
end
end

function m = localMango(n)
% CANlab "mango" split colormap materialized as an n x 3 LUT:
% purple (extreme -) -> green (near 0 -) | magenta (near 0 +) -> yellow (extreme +).
anchors = [.5 0 1; 0 .8 .3; 1 .2 1; 1 1 .3];
half = round(n/2);
neg = interp1([0 1], anchors(1:2,:), linspace(0,1,half));
pos = interp1([0 1], anchors(3:4,:), linspace(0,1,n-half));
m = [neg; pos];
end

function f = localScatter2D(XY, ptRGB, cats, catRGB, catIdx, ttl)
f = figure("Color","w","Position",[100 100 820 720]); hold on;
scatter(XY(:,1), XY(:,2), 60, ptRGB, "filled", "MarkerEdgeColor",[.25 .25 .25], "LineWidth",0.4);
h = gobjects(numel(cats),1);
for i = 1:numel(cats)
    h(i) = scatter(nan, nan, 60, catRGB(i,:), "filled");  % legend proxies
end
legend(h, strrep(cats,"_"," "), "Location","eastoutside", "Box","off", "FontSize",8);
box on; xlabel("dim 1"); ylabel("dim 2"); title(ttl);
% robust axis limits so a rare embedding outlier doesn't compress the cloud
lo = prctile(XY, 1); hi = prctile(XY, 99); pad = 0.08*(hi-lo) + eps;
xlim([lo(1)-pad(1), hi(1)+pad(1)]); ylim([lo(2)-pad(2), hi(2)+pad(2)]);
end

function localCategoryLegend(ax, cats, catRGB)
hold(ax, "on"); h = gobjects(numel(cats),1);
for i = 1:numel(cats)
    h(i) = plot(ax, nan, nan, "-", "Color", catRGB(i,:), "LineWidth", 3);
end
lg = legend(h, strrep(cats,"_"," "), "Location","eastoutside", "Box","off", "FontSize",7);
lg.Title.String = "category";
end

function localSave(fig, opts, name)
if ~opts.Save, return; end
base = fullfile(char(opts.OutDir), char(name));
try exportgraphics(fig, [base '.png'], "Resolution", 200); catch; end
try exportgraphics(fig, [base '.svg'], "ContentType", "vector"); catch; end
fprintf("  wrote %s.{png,svg}\n", base);
end
