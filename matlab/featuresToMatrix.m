function [X, info] = featuresToMatrix(ann, info, opts)
%FEATURESTOMATRIX One clip -> expanded numeric matrix [n x V] (no table).
%   [X, INFO] = FEATURESTOMATRIX(ANN) returns the fully-expanded numeric feature
%   matrix for one stimulus (every scalar channel one column, every vector channel one
%   column per component) plus the FEATUREINFO row describing each column. This is the
%   plain-matrix core of FEATURESTOTABLE: it skips building a wide MATLAB table, because
%   table operations scale terribly with column count — with the ~7.9k-variable set (the
%   4096-d Llama + 1024-d Qwen embeddings) the per-clip table round-trip was orders of
%   magnitude slower than the matrix. READANNOTATIONCORPUSFULL uses this directly.
%
%   [X, INFO] = FEATURESTOMATRIX(ANN, INFO) reuses a precomputed FEATUREINFO.
%
%   Name-value options (as in FEATURESTOTABLE):
%       "IncludeEmbeddings" (true)   include the opaque SigLIP/DINOv2/CLAP/Qwen/Llama dims
%       "NaNInapplicable"   (true)   channels with applicable=false -> all NaN
%
%   See also FEATURESTOTABLE, READANNOTATIONCORPUSFULL, FEATUREINFO.

arguments
    ann (1,1) struct
    info = []
    opts.IncludeEmbeddings (1,1) logical = true
    opts.NaNInapplicable (1,1) logical = true
end
if isempty(info), info = featureInfo(); end
keep = info.Numeric;
if ~opts.IncludeEmbeddings, keep = keep & ~info.IsEmbedding; end
info = info(keep, :);

n = numel(ann.time_sec);
X = nan(n, height(info));
[paths, ~, gidx] = unique(info.Path, "stable");        % fetch each channel dataset once
for gp = 1:numel(paths)
    rows = find(gidx == gp);
    ch = localGet(ann, paths(gp));
    if isempty(ch) || ~isfield(ch, "value") || ~isnumeric(ch.value), continue; end
    v = double(ch.value);
    applicable = ~isfield(ch, "applicable") || ch.applicable;
    if opts.NaNInapplicable && ~applicable, continue; end
    if isvector(v)
        if numel(v) == n, X(:, rows(1)) = v(:); end
    else
        if size(v,1) == n
            ci = info.CompIndex(rows);
            ok = ci <= size(v,2);
            X(:, rows(ok)) = v(:, ci(ok));
        end
    end
end
end

% -------------------------------------------------------------------------
function ch = localGet(ann, path)
% Walk ann.features by "a/b/c" path; return the channel struct or [].
ch = [];
node = ann.features;
parts = split(string(path), "/");
for i = 1:numel(parts)
    f = char(parts(i));
    if isstruct(node) && isfield(node, f)
        node = node.(f);
    else
        return
    end
end
if isstruct(node) && isfield(node, "value"), ch = node; end
end
