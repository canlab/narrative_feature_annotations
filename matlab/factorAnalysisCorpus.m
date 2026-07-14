function F = factorAnalysisCorpus(C, opts)
%FACTORANALYSISCORPUS Exploratory factor analysis of the feature corpus.
%   F = FACTORANALYSISCORPUS(C) runs EFA (MATLAB's FACTORAN, maximum-likelihood with
%   varimax rotation) on the expanded feature matrix from READANNOTATIONCORPUSFULL
%   and returns the factor solution plus a color-coded loadings plot.
%
%   By default it analyzes the INTERPRETABLE feature set — it excludes the opaque
%   SigLIP/DINOv2/CLAP embeddings and the two large fixed-taxonomy posteriors
%   (AudioSet 527 tags, Kinetics 400 actions), which otherwise dominate the solution
%   with hundreds of near-collinear columns. Set the options below to include them.
%
%   Missing data: features are z-scored, then the factor MODEL is fit from the
%   pairwise correlation matrix (using all available data per pair, so audio-only
%   clips still contribute their audio/language features) after a nearest
%   positive-definite adjustment. Per-timepoint FACTOR SCORES are then computed by
%   the regression method with mean-imputation (standardized NaN -> 0), so every
%   timepoint gets a score and the factors can be plotted as time series like the
%   raw features.
%
%   Name-value options:
%       "NumFactors"        (8)     number of factors to extract
%       "ExcludeEmbeddings" (true)  drop SigLIP/DINOv2/CLAP dimensions
%       "ExcludeTaxonomies" (true)  drop audioset_tags (527) + action_posteriors (400)
%       "Rotate"            ("varimax")  any FACTORAN rotation ("none","promax",...)
%       "MaxNaNFrac"        (0.8)   drop variables missing in > this fraction of rows
%                                   (0.8 keeps visual features, which are absent on the
%                                   ~67% of timepoints that come from audio-only clips)
%       "Plot"              (true)  draw the loadings heatmap + scree
%       "SavePng"           ("")    export the loadings figure to this path
%
%   Returns struct F with:
%       F.loadings   [p x m] rotated factor loadings
%       F.vars       [p x k] FEATUREINFO subset for the retained variables (row i <-> loadings row i)
%       F.scores     [T x m] per-timepoint factor scores (regression method)
%       F.specificVar[p x 1] uniquenesses; F.stats FACTORAN stats struct
%       F.eigs       eigenvalues of the correlation matrix (for the scree plot)
%       F.varExplained [1 x m] % variance each rotated factor accounts for
%       F.m, F.rotation, F.nEff
%
%   Example:
%       C = readAnnotationCorpusFull("annotations/corpus");
%       F = factorAnalysisCorpus(C, "NumFactors", 8);
%       % plot factor-3 time series for one clip:
%       r = C.stim=="BigBuckBunny"; plot(C.time_sec(r), F.scores(r,3));
%
%   See also READANNOTATIONCORPUSFULL, PLOTFEATUREMATRIX, FACTORAN, PCA.

arguments
    C (1,1) struct
    opts.NumFactors (1,1) double {mustBePositive, mustBeInteger} = 8
    opts.ExcludeEmbeddings (1,1) logical = true
    opts.ExcludeTaxonomies (1,1) logical = true
    opts.Rotate (1,1) string = "varimax"
    opts.MaxNaNFrac (1,1) double = 0.8
    opts.Plot (1,1) logical = true
    opts.SavePng (1,1) string = ""
end

info = C.info; X = C.X;

% ---- column selection ----
keep = info.Numeric;
if opts.ExcludeEmbeddings, keep = keep & ~info.IsEmbedding; end
if opts.ExcludeTaxonomies
    keep = keep & ~ismember(info.Leaf, ["audioset_tags", "action_posteriors"]);
end
keep = keep & (mean(isnan(X),1)' <= opts.MaxNaNFrac);      % not-too-missing
X = X(:, keep); info = info(keep, :);

% z-score; drop zero-variance columns
mu = mean(X,1,"omitnan"); sd = std(X,0,1,"omitnan");
good = sd > 0 & isfinite(sd);
X = X(:, good); info = info(good,:); mu = mu(good); sd = sd(good);
Z = (X - mu) ./ sd;

% ---- correlation matrix (pairwise) + nearest-PD repair ----
R = corr(Z, "rows", "pairwise");
R(~isfinite(R)) = 0; R(1:size(R,1)+1:end) = 1;
% drop columns still fully collinear (|r|=1 with an earlier column)
[R, info, Z] = dropCollinear(R, info, Z);
R = nearestPD(R);
p = size(R,1);
nEff = size(Z,1);

m = min(opts.NumFactors, p-1);
if m < opts.NumFactors
    warning("factorAnalysisCorpus:m", "Reduced NumFactors to %d (only %d variables).", m, p);
end

% ---- fit factor model ----
% factoran needs complete data; audio-only clips make visual columns NaN, so when
% any NaNs are present we fit from the (repaired) correlation matrix instead.
if any(isnan(Z(:)))
    [L, psi, ~, stats] = factoran(R, m, "Xtype","covariance", "Nobs",nEff, ...
                                  "Rotate",char(opts.Rotate));
else
    [L, psi, ~, stats] = factoran(Z, m, "Xtype","data", "Rotate",char(opts.Rotate), ...
                                  "Maxit",1000);
end

% ---- per-timepoint factor scores (regression method, mean-imputed) ----
Zi = Z; Zi(isnan(Zi)) = 0;                 % standardized mean-imputation
W = R \ L;                                 % regression weights  (p x m)
scores = Zi * W;

% variance explained by each rotated factor (sum of squared loadings / p)
ve = 100 * sum(L.^2, 1) / p;
[ve, si] = sort(ve, "descend"); L = L(:, si); scores = scores(:, si);

F.loadings = L; F.vars = info; F.scores = scores; F.specificVar = psi;
F.stats = stats; F.varExplained = ve; F.m = m; F.rotation = string(opts.Rotate); F.nEff = nEff;
F.eigs = sort(eig(R), "descend");

fprintf("EFA: %d variables, %d factors (%s), n_eff=%d. Variance explained: %s%%\n", ...
    p, m, opts.Rotate, nEff, join(compose("%.1f", ve), "/"));

if opts.Plot, plotLoadings(F, opts.SavePng); end
end

% -------------------------------------------------------------------------
function [R, info, Z] = dropCollinear(R, info, Z)
p = size(R,1); keep = true(p,1);
for i = 2:p
    if any(abs(R(i, 1:i-1)) > 0.999 & keep(1:i-1)'), keep(i) = false; end
end
R = R(keep,keep); info = info(keep,:); Z = Z(:,keep);
end

function A = nearestPD(A)
A = (A + A')/2;
[V,D] = eig(A); d = diag(D);
d(d < 1e-6) = 1e-6;
A = V*diag(d)*V';
A = (A + A')/2;
s = sqrt(diag(A)); A = A ./ (s*s');       % rescale to unit diagonal (correlation)
A = (A + A')/2;
A(1:size(A,1)+1:end) = 1;                 % exact unit diagonal
end

function plotLoadings(F, savePng)
fig = figure("Color","w","Position",[100 100 980 760]);
% sort variables by class then by dominant factor for a readable block structure
classes = ["visual","audio","language","social","situation","affect"];
key = double(arrayfun(@(c) find(classes==c,1), F.vars.Class));
[~, dom] = max(abs(F.loadings), [], 2);
[~, ord] = sortrows([key, dom]);
L = F.loadings(ord,:); vinfo = F.vars(ord,:);

ax = axes(fig, "Position",[0.30 0.08 0.60 0.84]);
imagesc(ax, L); set(ax, "CLim",[-1 1]);
colormap(ax, localDiv()); cb = colorbar(ax); cb.Label.String = "loading";
ax.XTick = 1:F.m; ax.XTickLabel = compose("F%d", 1:F.m);
xlabel(ax, sprintf("factors (%s rotation)", F.rotation));
title(ax, sprintf("EFA loadings — %d variables x %d factors", size(L,1), F.m));
% label a subset of rows (too many to show all)
step = max(1, round(size(L,1)/60));
ax.YTick = 1:step:size(L,1);
lab = vinfo.Leaf + string(compose(" [%s]", extractBefore(vinfo.Level+"   ",4)));
comp = vinfo.Component; hasC = comp~="";
lab(hasC) = vinfo.Leaf(hasC) + ":" + comp(hasC);
ax.YTickLabel = lab(1:step:end); ax.FontSize = 7;

% class color strip
strip = axes(fig, "Position",[0.27 0.08 0.02 0.84]);
image(strip, reshape(vinfo.Color,[height(vinfo) 1 3]));
strip.XTick=[]; strip.YTick=[]; box(strip,"on"); linkaxes([ax strip],"y");

% scree inset
sc = axes(fig, "Position",[0.06 0.62 0.16 0.30]);
plot(sc, F.eigs(1:min(20,numel(F.eigs))), "-o","MarkerSize",3); grid(sc,"on");
title(sc,"scree"); xlabel(sc,"factor"); ylabel(sc,"eigenvalue"); sc.FontSize=7;

if savePng ~= ""
    exportgraphics(fig, savePng, "Resolution",150); fprintf("wrote %s\n", savePng);
end
end

function m = localDiv()
n=256; x=linspace(0,1,n)'; m=[min(1,1.8*x), 1-abs(2*x-1), min(1,1.8*(1-x))];
end
