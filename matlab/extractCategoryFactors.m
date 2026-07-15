function C = extractCategoryFactors(C, opts)
%EXTRACTCATEGORYFACTORS Reduce each model/category to a few factors and store them.
%   C = EXTRACTCATEGORYFACTORS(C) takes the FULL expanded corpus from
%   READANNOTATIONCORPUSFULL and runs factor analysis SEPARATELY within each category:
%   each of the 13 multivariate model outputs (SigLIP embedding, DINOv2 embedding,
%   CLAP embedding, AudioSet tags, action posteriors, EmoNet, GoEmotions text-emotion,
%   SigLIP/CLAP probes, MFCC, chroma, facial affect, text sentiment) AND the block of
%   interpretable scalar features. For every category it extracts factor-score time
%   series (one per factor, over all timepoints) and attaches them to C as
%   C.extracted_factors, so the ~2.7k raw variables collapse to a compact, labeled set
%   of per-model factors usable as fMRI regressors.
%
%   Method: features are z-scored; the factor model is fit from the pairwise
%   correlation matrix (nearest-positive-definite repaired), so audio-only clips still
%   contribute their audio/language factors. FACTORAN (max-likelihood, varimax) is used
%   where the block's degrees of freedom allow; otherwise it falls back to PCA (typical
%   for the high-dimensional opaque embeddings). Per-timepoint scores use the regression
%   method with mean-imputation, so every timepoint gets a score.
%
%   Name-value options:
%       "NumFactors" (10)     target factors per category (capped by block dof/rank)
%       "Rotate"     ("varimax")  rotation for the FACTORAN path
%       "Method"     ("auto")  "auto" | "factoran" | "pca"
%       "Save"       ("")      if set, save the extracted_factors struct to this .mat
%       "Verbose"    (true)
%
%   Adds C.extracted_factors with fields:
%       .scores      [T x F] all factor scores concatenated across categories
%       .labels      [F x k] table: FactorName, Category, Model, Class, Method, VarExpl
%       .byCategory  struct array, one per category, with:
%                    name, model, class, kind, nVars, nFactors, method,
%                    loadings [nVars x m], varExplained [1 x m], scores [T x m],
%                    factorNames [1 x m], vars (FEATUREINFO subset)
%       .stim, .time_sec, .ids           row keys (same rows as C.X)
%
%   Example:
%       C = readAnnotationCorpusFull("annotations/corpus");
%       C = extractCategoryFactors(C, "NumFactors", 10, "Save", "analysis/extracted_factors.mat");
%       % SigLIP visual factors as a time series for one clip:
%       r = C.stim=="BigBuckBunny"; s = C.extracted_factors.byCategory;
%       si = s(strcmp({s.name},"siglip_embedding"));
%       plot(C.time_sec(r), si.scores(r,1:3));
%
%   See also READANNOTATIONCORPUSFULL, FACTORANALYSISCORPUS, FEATUREINFO.

arguments
    C (1,1) struct
    opts.NumFactors (1,1) double {mustBePositive, mustBeInteger} = 10
    opts.Rotate (1,1) string = "varimax"
    opts.Method (1,1) string {mustBeMember(opts.Method, ["auto","factoran","pca"])} = "auto"
    opts.Save (1,1) string = ""
    opts.Verbose (1,1) logical = true
end

info = C.info; X = C.X; T = size(X,1);

% ---- define categories: the 13 multivariate channels + the interpretable-scalar block
vecLeaves = unique(info.Leaf(info.Dtype == "vector"), "stable");
cats = struct("name",{}, "model",{}, "class",{}, "kind",{}, "cols",{});
for i = 1:numel(vecLeaves)
    
    idx = find(info.Leaf == vecLeaves(i));
    kind = "interpretable-vector";
    if info.IsEmbedding(idx(1)), kind = "embedding";
    elseif ismember(vecLeaves(i), ["audioset_tags","action_posteriors"]), kind = "taxonomy"; end
    
    cats(end+1) = struct("name", vecLeaves(i), "model", info.Model(idx(1)), ...
        "class", info.Class(idx(1)), "kind", kind, "cols", idx(:)'); %#ok<AGROW>
end

sIdx = find(info.Dtype ~= "vector" & info.Numeric);

cats(end+1) = struct("name","interpretable_scalars", "model","(mixed)", ...
    "class","(mixed)", "kind","interpretable-scalar", "cols", sIdx(:)');

% ---- run FA/PCA per category
byCat = struct([]); allScores = []; FN=strings(0,1); Cat=strings(0,1);
Mdl=strings(0,1); Cls=strings(0,1); Meth=strings(0,1); VE=zeros(0,1);

for i = 1:numel(cats)

    ca = cats(i);
    cols = ca.cols;
    sub = X(:, cols);
    vinfo = info(cols, :);

    % z-score; drop zero-variance columns
    mu = mean(sub,1,"omitnan"); sd = std(sub,0,1,"omitnan");
    keep = sd > 0 & isfinite(sd);
    sub = sub(:, keep); vinfo = vinfo(keep,:); mu = mu(keep); sd = sd(keep);
    Z = (sub - mu) ./ sd;
    
    d = size(Z,2);
    if d == 0, continue; end

    % pairwise correlation -> nearest PD; drop fully-collinear columns
    R = corr(Z, "rows","pairwise"); R(~isfinite(R)) = 0; R(1:d+1:end) = 1;
    [R, vinfo, Z] = localDropCollinear(R, vinfo, Z);
    R = localNearestPD(R);
    d = size(R,1);
    nEff = size(Z,1);

    m = min(opts.NumFactors, max(1, d-1));
    method = opts.Method;
    if method == "auto"
        method = "factoran";
        if d < 3 || m > localMaxFactoranM(d), method = "pca"; end
    end

    Zi = Z; Zi(isnan(Zi)) = 0;                      % standardized mean-imputation
    if method == "factoran" && m > localMaxFactoranM(d)
        m = max(1, localMaxFactoranM(d));
    end
    try
        if method == "factoran"
            [L, ~, ~, ~] = factoran(R, m, "Xtype","covariance", "Nobs",nEff, "Rotate",char(opts.Rotate));
            W = R \ L;
            scores = Zi * W;
            ve = 100 * sum(L.^2,1) / d;
        else
            error("use pca");                       % jump to catch/pca path
        end
    catch
        method = "pca";
        [V, ev] = localTopEig(R, m);
        L = V .* sqrt(max(ev,0))';                  % component loadings
        scores = Zi * V;                            % PC scores on standardized data
        ve = 100 * ev' / d;
    end

    % order factors by variance explained (desc)
    [ve, si] = sort(ve, "descend"); L = L(:,si); scores = scores(:,si);
    fnames = ca.name + "_F" + string(1:m);

    e = numel(byCat) + 1;
    byCat(e).name = ca.name; byCat(e).model = ca.model; byCat(e).class = ca.class;
    byCat(e).kind = ca.kind; byCat(e).nVars = d; byCat(e).nFactors = m;
    byCat(e).method = string(method); byCat(e).loadings = L; byCat(e).varExplained = ve;
    byCat(e).scores = scores; byCat(e).factorNames = fnames; byCat(e).vars = vinfo;

    allScores = [allScores, scores]; %#ok<AGROW>
    FN=[FN; fnames(:)]; Cat=[Cat; repmat(ca.name,m,1)]; Mdl=[Mdl; repmat(string(ca.model),m,1)];
    Cls=[Cls; repmat(string(ca.class),m,1)]; Meth=[Meth; repmat(string(method),m,1)]; VE=[VE; ve(:)];

    if opts.Verbose
        fprintf("  %-22s %-9s d=%4d -> %2d factors (%s)  var%%=%s\n", ...
            ca.name, "["+ca.kind+"]", d, m, method, join(compose("%.0f",ve),"/"));
    end
end

EF.scores = allScores;
EF.labels = table(FN, Cat, Mdl, Cls, Meth, VE, 'VariableNames', ...
    {'FactorName','Category','Model','Class','Method','VarExplained'});
EF.byCategory = byCat;
EF.stim = C.stim; EF.time_sec = C.time_sec; EF.ids = C.ids;
C.extracted_factors = EF;

if opts.Verbose
    fprintf("Extracted %d factors across %d categories (%d timepoints).\n", ...
        size(allScores,2), numel(byCat), T);
end
if opts.Save ~= ""
    extracted_factors = EF;
    save(opts.Save, "extracted_factors", "-v7.3");
    fprintf("wrote %s\n", opts.Save);
end
end

% -------------------------------------------------------------------------
function [R, info, Z] = localDropCollinear(R, info, Z)
p = size(R,1); keep = true(p,1);
for i = 2:p
    if any(abs(R(i,1:i-1)) > 0.999 & keep(1:i-1)'), keep(i) = false; end
end
R = R(keep,keep); info = info(keep,:); Z = Z(:,keep);
end

function A = localNearestPD(A)
A = (A + A')/2; [V,D] = eig(A); d = diag(D); d(d < 1e-6) = 1e-6;
A = V*diag(d)*V'; A = (A + A')/2; s = sqrt(diag(A)); A = A ./ (s*s');
A = (A + A')/2; A(1:size(A,1)+1:end) = 1;
end

function [V, ev] = localTopEig(R, m)
[V,D] = eig((R+R')/2); ev = diag(D);
[ev, ix] = sort(ev, "descend"); V = V(:, ix);
m = min(m, size(V,2)); V = V(:,1:m); ev = ev(1:m);
end

function mmax = localMaxFactoranM(d)
% largest m>=1 satisfying FACTORAN's dof rule (d-m)^2 >= d+m
mmax = 0;
for m = 1:d-1
    if (d-m)^2 >= d+m, mmax = m; end
end
end
