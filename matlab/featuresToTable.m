function [T, info] = featuresToTable(ann, info, opts)
%FEATURESTOTABLE Expand ALL feature channels of one stimulus into a wide table.
%   T = FEATURESTOTABLE(ANN) returns a timetable with one row per timepoint (on the
%   common grid) and one column per *variable* in the fully-expanded feature set:
%   every scalar channel becomes one column and every vector channel is expanded
%   into one column per component (SigLIP -> 768 cols, action -> 400, EmoNet -> 20,
%   ...). Columns follow FEATUREINFO order exactly, so tables from different clips
%   are column-compatible and can be vertically concatenated. Unlike
%   FEATURESTOTIMETABLE (scalars only), this keeps the full ~2.7k-variable set.
%
%   [T, INFO] = FEATURESTOTABLE(ANN) also returns the FEATUREINFO label table whose
%   row i describes column i of T.
%
%   T = FEATURESTOTABLE(ANN, INFO) reuses a precomputed FEATUREINFO (faster in a
%   loop). Pass [] to build it internally.
%
%   Name-value options:
%       "IncludeEmbeddings" (true)  include the opaque SigLIP/DINOv2/CLAP dims
%       "NaNInapplicable"   (true)  channels with applicable=false -> all NaN
%
%   Only numeric variables (INFO.Numeric==true) are included; categorical "*_top"
%   and free-text labels are left out (their codes are not magnitudes).
%
%   See also FEATUREINFO, FEATURESTOTIMETABLE, READANNOTATIONCORPUSFULL.

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
V = height(info);
X = nan(n, V);

% Group rows by channel path so each channel dataset is fetched once.
[paths, ~, gidx] = unique(info.Path, "stable");
for gp = 1:numel(paths)
    rows = find(gidx == gp);                 % info rows (columns of X) for this channel
    ch = localGet(ann, paths(gp));
    if isempty(ch) || ~isfield(ch, "value") || ~isnumeric(ch.value), continue; end
    v = double(ch.value);
    applicable = ~isfield(ch, "applicable") || ch.applicable;
    if opts.NaNInapplicable && ~applicable, continue; end     % leave as NaN
    if isvector(v)                           % scalar channel [n x 1]
        if numel(v) == n, X(:, rows(1)) = v(:); end
    else                                     % vector channel [n x D]
        if size(v,1) == n
            ci = info.CompIndex(rows);        % component index per target column
            ok = ci <= size(v,2);
            X(:, rows(ok)) = v(:, ci(ok));
        end
    end
end

rowTimes = seconds(ann.time_sec(:));
T = array2timetable(X, "RowTimes", rowTimes, "VariableNames", cellstr(info.VarName));
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
