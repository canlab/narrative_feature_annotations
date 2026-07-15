function A = plot_tsne_umap_annotations(opts)
%PLOT_TSNE_UMAP_ANNOTATIONS t-SNE and UMAP of all ~2,700 feature annotations.
%   A = PLOT_TSNE_UMAP_ANNOTATIONS() loads the full expanded feature matrix
%   (READANNOTATIONCORPUSFULL) and embeds the FEATURE VARIABLES (not the timepoints)
%   into 2-D with t-SNE and UMAP, so each point is one of the ~2,768 annotation
%   variables and nearby points behave similarly over time. Points are colored by
%   feature category (the six domains of the feature summary table: visual, audio,
%   language, social, situation, affect). Saves an .svg and .png of each embedding to
%   matlab/figures/ and returns a struct A with the embeddings and labels.
%
%   Each variable is z-scored over time (NaN from inapplicable clips -> 0 after
%   standardizing), reduced to 50 principal components, then embedded; two variables
%   are near if their time courses covary, so the layout reveals how the annotation
%   space is organized by modality.
%
%   Name-value options:
%       "OutDir"     ("matlab/figures")
%       "CanlabPath" ("/Users/f003vz1/Documents/GitHub/CanlabCore/CanlabCore")
%       "NumPCA"     (50)     PCs fed to t-SNE / UMAP
%       "Perplexity" (30)     t-SNE perplexity
%       "NNeighbors" (30)     UMAP n_neighbors
%       "Save"       (true)   write .svg + .png
%
%   See also PLOTFACTORSCORES, READANNOTATIONCORPUSFULL, FEATUREINFO, RUN_UMAP.

arguments
    opts.OutDir (1,1) string = "matlab/figures"
    opts.CanlabPath (1,1) string = "/Users/f003vz1/Documents/GitHub/CanlabCore/CanlabCore"
    opts.NumPCA (1,1) double = 50
    opts.Perplexity (1,1) double = 30
    opts.NNeighbors (1,1) double = 30
    opts.Save (1,1) logical = true
end

if isfolder(opts.CanlabPath), addpath(genpath(char(opts.CanlabPath))); end
if ~isfolder(opts.OutDir), mkdir(opts.OutDir); end

C = readAnnotationCorpusFull("annotations/corpus", "Verbose", false);
X = C.X; info = C.info;                       % X: [T x V], V ~ 2768
V = size(X, 2);

% ---- per-variable standardize over time, impute, reduce to PCs ----
mu = mean(X, 1, "omitnan"); sd = std(X, 0, 1, "omitnan");
Z = (X - mu) ./ sd;                           % [T x V]
Z(~isfinite(Z)) = 0;                          % inapplicable / constant -> 0
Zv = Z.';                                     % [V x T]: one row per variable
npc = min(opts.NumPCA, V-1);
[~, pcs] = pca(Zv, "NumComponents", npc);     % [V x npc]

% ---- color each variable by its feature-class domain ----
classes = ["visual","audio","language","social","situation","affect"];
classRGB = [ .39 .40 .95;  .02 .71 .83;  .96 .62 .04;  .93 .28 .60;  .06 .73 .51;  .94 .27 .27];
cidx = zeros(V,1);
for i = 1:numel(classes), cidx(info.Class == classes(i)) = i; end
ptRGB = classRGB(max(cidx,1), :);
present = ismember(1:numel(classes), unique(cidx))';   % classes actually present
A.info = info; A.class_names = classes; A.class_colors = classRGB;

fprintf("plot_tsne_umap_annotations: %d variables, %d PCs.\n", V, npc);
counts = arrayfun(@(k) sum(cidx==k), 1:numel(classes));

% ===================================================================================
% t-SNE
% ===================================================================================
rng(7);
A.tsne = tsne(pcs, "Perplexity", opts.Perplexity, "Standardize", true);
f1 = localScatter(A.tsne, ptRGB, classes, classRGB, counts, present, ...
    sprintf("t-SNE of %d feature annotations (perplexity %d)", V, opts.Perplexity));
localSave(f1, opts, "07_tsne_annotations");

% ===================================================================================
% UMAP
% ===================================================================================
try
    rng(7); close all;                        % umap toolbox needs a clean figure state
    A.umap = run_umap(pcs, 'method', 'Java', 'n_neighbors', opts.NNeighbors, ...
        'min_dist', 0.3, 'verbose', 'none');
    close all;
    f2 = localScatter(A.umap, ptRGB, classes, classRGB, counts, present, ...
        sprintf("UMAP of %d feature annotations", V));
    localSave(f2, opts, "08_umap_annotations");
catch ME
    warning("plot_tsne_umap_annotations:umap", "UMAP skipped (%s).", ME.message);
end

fprintf("Done. Figures saved to %s.\n", opts.OutDir);
end

% -------------------------------------------------------------------------
function f = localScatter(XY, ptRGB, classes, classRGB, counts, present, ttl)
f = figure("Color","w","Position",[100 100 900 740]); hold on;
scatter(XY(:,1), XY(:,2), 16, ptRGB, "filled", "MarkerFaceAlpha", 0.65, ...
    "MarkerEdgeColor", "none");
h = gobjects(sum(present),1); labs = strings(sum(present),1); j = 0;
for i = 1:numel(classes)
    if ~present(i), continue; end
    j = j + 1;
    h(j) = scatter(nan, nan, 60, classRGB(i,:), "filled");   % legend proxy
    labs(j) = sprintf("%s (%d)", classes(i), counts(i));
end
lg = legend(h, labs, "Location","eastoutside", "Box","off", "FontSize",10);
lg.Title.String = "feature category";
box on; xlabel("dim 1"); ylabel("dim 2"); title(ttl);
lo = prctile(XY,1); hi = prctile(XY,99); pad = 0.06*(hi-lo) + eps;
xlim([lo(1)-pad(1), hi(1)+pad(1)]); ylim([lo(2)-pad(2), hi(2)+pad(2)]);
end

function localSave(fig, opts, name)
if ~opts.Save, return; end
base = fullfile(char(opts.OutDir), char(name));
try exportgraphics(fig, [base '.png'], "Resolution", 200); catch; end
try exportgraphics(fig, [base '.svg'], "ContentType", "vector"); catch; end
fprintf("  wrote %s.{png,svg}\n", base);
end
