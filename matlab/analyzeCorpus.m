function res = analyzeCorpus(C, opts)
%ANALYZECORPUS Structure analysis of the annotation corpus: correlation, PCA, network.
%   RES = ANALYZECORPUS(C) takes the struct from READANNOTATIONCORPUS and produces:
%     - a clustered cross-feature correlation heatmap,
%     - a PCA scree / cumulative-variance plot,
%     - a channel-level correlation network graph (nodes colored by feature class),
%     - a feature-class-level network (mean |r| between classes),
%   and returns RES with fields: channels, class, R (corr), order (cluster order),
%   coeff/score/explained (PCA), classR (class x class mean |r|), classes.
%
%   Options:
%     "Threshold"  edge threshold on |r| for the channel network (default 0.30)
%     "MaxNaNFrac" drop channels with more than this NaN fraction (default 0.40)
%     "OutDir"     folder to save figure PNGs (default "analysis/figures")
%     "Save"       save PNGs (default true)
%
%   See also READANNOTATIONCORPUS.

arguments
    C (1,1) struct
    opts.Threshold (1,1) double = 0.30
    opts.MaxNaNFrac (1,1) double = 0.40
    opts.OutDir (1,1) string = "analysis/figures"
    opts.Save (1,1) logical = true
end

% ---- drop mostly-NaN channels, then z-score (NaN-aware) -------------------
nanfrac = mean(isnan(C.X), 1);
keep = nanfrac <= opts.MaxNaNFrac;
X = C.X(:, keep);
names = C.channels(keep);
cls = extractBefore(names, "__");
mu = mean(X, 1, "omitnan");               % NaN-aware z-score (normalize() keeps NaN)
sigma = std(X, 0, 1, "omitnan");
sigma(sigma == 0 | isnan(sigma)) = 1;
Xz = (X - mu) ./ sigma;
Xz(isnan(Xz)) = 0;                        % impute the genuinely-missing entries at the mean

% ---- correlation + clustering --------------------------------------------
R = corr(X, "rows", "pairwise");
R(isnan(R)) = 0;                          % constant/insufficient-overlap channels -> 0
R(1:size(R, 1) + 1:end) = 1;              % force unit diagonal so squareform() is valid
D = 1 - abs(R);
order = optimalleaforder(linkage(squareform(D, "tovector"), "average"), ...
    squareform(D, "tovector"));

% ---- PCA ------------------------------------------------------------------
[coeff, score, ~, ~, explained] = pca(Xz);

% ---- class-level network --------------------------------------------------
[classes, ~, ci] = unique(cls, "stable");
nc = numel(classes);
classR = zeros(nc);
for a = 1:nc
    for b = 1:nc
        block = abs(R(ci == a, ci == b));
        if a == b
            block(logical(eye(sum(ci == a)))) = NaN;   % exclude self-corr
        end
        classR(a, b) = mean(block(:), "omitnan");
    end
end

res = struct("channels", names, "class", cls, "R", R, "order", order, ...
    "coeff", coeff, "score", score, "explained", explained, ...
    "classes", classes, "classR", classR);

% ========================= figures ========================================
if opts.Save && ~isfolder(opts.OutDir)
    mkdir(opts.OutDir);
end
cmap = lines(nc);

% (1) clustered correlation heatmap
f1 = figure("Color", "w", "Position", [100 100 760 680]);
imagesc(R(order, order), [-1 1]);
axis square; colorbar; colormap(f1, redbluemap());
title(sprintf("Cross-feature correlation (%d channels, %d timepoints)", ...
    numel(names), size(X, 1)));
set(gca, "XTick", [], "YTick", 1:numel(names), "YTickLabel", names(order), ...
    "TickLabelInterpreter", "none", "FontSize", 6);
saveFig(f1, opts, "corr_heatmap.png");

% (2) PCA scree + cumulative
f2 = figure("Color", "w", "Position", [100 100 720 320]);
tiledlayout(1, 2, "Padding", "compact");
nexttile; bar(explained(1:min(20, end))); xlabel("PC"); ylabel("% variance");
title("Scree (top 20 PCs)");
nexttile; plot(cumsum(explained), "-o", "MarkerSize", 3); xlabel("PC #");
ylabel("cumulative % variance"); yline(80, "r--"); ylim([0 100]); grid on;
title("Cumulative variance explained");
saveFig(f2, opts, "pca_scree.png");

% (3) channel network graph (nodes colored by class)
A = (abs(R) >= opts.Threshold) & ~eye(numel(names));
G = graph(A, cellstr(names), "omitselfloops");
f3 = figure("Color", "w", "Position", [100 100 820 760]);
ax3 = axes(f3);
p = plot(ax3, G, "Layout", "force", "UseGravity", true, "MarkerSize", 5, "NodeLabel", {});
hold(ax3, "on");
for k = 1:nc
    highlight(p, find(ci == k), "NodeColor", cmap(k, :));
end
lg = gobjects(1, nc);
for k = 1:nc
    lg(k) = scatter(ax3, nan, nan, 36, cmap(k, :), "filled");
end
legend(ax3, lg, classes, "Location", "eastoutside", "Interpreter", "none");
title(ax3, sprintf("Feature correlation network (|r| \\geq %.2f)", opts.Threshold));
axis(ax3, "off");
hold(ax3, "off");
saveFig(f3, opts, "feature_network.png");

% (4) class-level network
f4 = figure("Color", "w", "Position", [100 100 640 600]);
Acl = classR; Acl(logical(eye(nc))) = 0;
Gc = graph(Acl, cellstr(classes), "upper", "omitselfloops");
w = Gc.Edges.Weight;
pc = plot(Gc, "Layout", "circle", "LineWidth", 1 + 6 * w / max(w), ...
    "NodeColor", cmap, "MarkerSize", 10, "NodeFontSize", 11, ...
    "EdgeAlpha", 0.7, "Interpreter", "none");
labeledge(pc, Gc.Edges.EndNodes(:, 1), Gc.Edges.EndNodes(:, 2), ...
    compose("%.2f", w));
title("Feature-class network (edge = mean |r| between classes)");
axis off;
saveFig(f4, opts, "class_network.png");

fprintf("Analysis done: %d channels kept, PC1-5 explain %.0f%%. Figures -> %s\n", ...
    numel(names), sum(explained(1:min(5, end))), opts.OutDir);
end

% -------------------------------------------------------------------------
function saveFig(f, opts, name)
if opts.Save
    exportgraphics(f, fullfile(opts.OutDir, name), "Resolution", 130);
end
end

function cm = redbluemap()
n = 64; r = [linspace(0, 1, n/2) ones(1, n/2)]';
g = [linspace(0, 1, n/2) linspace(1, 0, n/2)]';
b = [ones(1, n/2) linspace(1, 0, n/2)]';
cm = [r g b];
end
