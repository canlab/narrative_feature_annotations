function ann = readAnnotations(filepath)
%READANNOTATIONS Load a Narrative Feature Extraction annotation into MATLAB.
%   ANN = READANNOTATIONS(FILEPATH) reads the canonical HDF5 (.h5) file, a JSON
%   pure-profile file (.json), or a stimulus folder (auto-finds the .h5), and
%   returns a struct with fields:
%       .stimulus   - struct of stimulus metadata
%       .time       - struct (rate_hz, t_start_sec, n_samples, bin_reference)
%       .time_sec   - column vector of timepoints on the common grid
%       .features   - nested struct mirroring the hierarchy; each leaf channel
%                     has a .value field (numeric scalar/vector, or cellstr for
%                     labels) plus metadata; JSON null / HDF5 fill -> NaN
%       .provenance - pass-through where present
%
%   Format spec: docs/design/ANNOTATION_FORMAT.md §7. Companion helpers:
%       getFeature(ANN,'visual/low_level_static/luminance')  -> channel struct
%       featuresToTimetable(ANN)                              -> timetable of scalars
%
%   See also GETFEATURE, FEATURESTOTIMETABLE.

arguments
    filepath (1,1) string
end

if isfolder(filepath)
    hits = dir(fullfile(filepath, "*.h5"));
    if isempty(hits)
        error("readAnnotations:noH5", "No .h5 file found in folder %s.", filepath);
    end
    filepath = string(fullfile(hits(1).folder, hits(1).name));
end
if ~isfile(filepath)
    error("readAnnotations:notFound", "File not found: %s.", filepath);
end

[~, ~, ext] = fileparts(filepath);
switch lower(ext)
    case ".h5"
        ann = readH5(filepath);
    case ".json"
        ann = readJSON(filepath);
    otherwise
        error("readAnnotations:badExt", "Unsupported extension '%s'.", ext);
end
end

% ========================== HDF5 (canonical) =============================
function ann = readH5(filepath)
ann = struct();
ann.stimulus = attrStruct(filepath, "/stimulus");
ann.time = attrStruct(filepath, "/time");
ann.time_sec = double(h5read(filepath, "/time/time_sec"));
ann.time_sec = ann.time_sec(:);
ann.features = walkGroup(filepath, h5info(filepath, "/features"), numel(ann.time_sec));
end

function g = walkGroup(filepath, ginfo, n)
g = struct();
% leaf datasets first
for i = 1:numel(ginfo.Datasets)
    ds = ginfo.Datasets(i);
    key = lastName(ds.Name);
    g.(key) = readChannel(filepath, [ginfo.Name '/' ds.Name], ds, n);
end
% subgroups (attach *__onsets helper groups to their channel)
for i = 1:numel(ginfo.Groups)
    sub = ginfo.Groups(i);
    name = lastName(sub.Name);
    if endsWith(name, "__onsets")
        base = extractBefore(name, "__onsets");
        if isfield(g, base)
            g.(base).onsets = double(h5read(filepath, [sub.Name '/time_sec']));
        end
    else
        g.(name) = walkGroup(filepath, sub, n);
    end
end
end

function ch = readChannel(filepath, fullpath, ds, n)
val = h5read(filepath, string(fullpath));
if isnumeric(val)
    val = squeeze(val);
    % Python always stores time-first [n x D]; h5read returns it transposed [D x n],
    % so always transpose 2-D datasets (the previous size-based heuristic mis-oriented
    % vectors whose dim happened to equal the grid length).
    if isvector(val)
        val = val(:);
    else
        val = val.';
    end
    val = double(val);
    val(val == -1 & isAttr(ds, "dtype", "bool")) = NaN;  % bool fill -> NaN
end
ch.value = val;
for a = 1:numel(ds.Attributes)
    at = ds.Attributes(a);
    name = matlab.lang.makeValidName(at.Name);
    ch.(name) = at.Value;
end
if isfield(ch, "applicable")
    a = ch.applicable;                 % may be numeric, or an enum cell {'TRUE'}
    if iscell(a), a = a{1}; end
    if ischar(a) || isstring(a)
        ch.applicable = any(strcmpi(string(a), ["TRUE", "1"]));
    else
        ch.applicable = logical(a);
    end
end
end

function tf = isAttr(ds, name, want)
tf = false;
for a = 1:numel(ds.Attributes)
    if strcmp(ds.Attributes(a).Name, name)
        tf = isequal(string(ds.Attributes(a).Value), string(want));
        return;
    end
end
end

function s = attrStruct(filepath, grouppath)
info = h5info(filepath, grouppath);
s = struct();
for a = 1:numel(info.Attributes)
    s.(matlab.lang.makeValidName(info.Attributes(a).Name)) = info.Attributes(a).Value;
end
end

function nm = lastName(path)
parts = split(string(path), "/");
nm = char(parts(end));
end

% ============================ JSON profile ===============================
function ann = readJSON(filepath)
ann = jsondecode(fileread(filepath));
tl = ann.timeline;
ann.time = struct("rate_hz", tl.rate_hz, "t_start_sec", tl.t_start_sec, ...
    "n_samples", tl.n_samples, "bin_reference", "center");   % mirror the .h5 layout
if isfield(tl, "time_sec") && ~isempty(tl.time_sec)
    ann.time_sec = tl.time_sec(:);
else
    nn = double(tl.n_samples);
    ann.time_sec = tl.t_start_sec + (0:nn-1)' / tl.rate_hz;
end
if isfield(ann, "features")
    ann.features = normalizeGroup(ann.features);
end
end

function g = normalizeGroup(g)
fn = fieldnames(g);
for i = 1:numel(fn)
    node = g.(fn{i});
    if isstruct(node) && isfield(node, "value")
        g.(fn{i}) = normalizeChannel(node);
    elseif isstruct(node)
        g.(fn{i}) = normalizeGroup(node);
    end
end
end

function ch = normalizeChannel(ch)
v = ch.value;
if iscell(v)
    isNum = cellfun(@(x) isnumeric(x) && isscalar(x), v);
    if all(isNum | cellfun(@isempty, v))
        out = nan(numel(v), 1);
        out(isNum) = cell2mat(v(isNum));
        ch.value = out;
    end
end
end
