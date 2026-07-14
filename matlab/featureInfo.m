function info = featureInfo(templatePath)
%FEATUREINFO Metadata/label table for every expanded feature variable.
%   INFO = FEATUREINFO() reads schema/channel_template.json and returns one row
%   per *variable* in the fully-expanded feature set: every scalar channel is one
%   row, and every vector channel is expanded into one row per component (so the
%   768-D SigLIP embedding contributes 768 rows, EmoNet 20 rows, etc.). This is the
%   companion label table for the wide feature matrix built by FEATURESTOTABLE /
%   READANNOTATIONCORPUSFULL — INFO row i describes column i of that matrix.
%
%   INFO = FEATUREINFO(TEMPLATEPATH) uses a specific channel template.
%
%   Columns of INFO:
%       VarName     valid, unique MATLAB name used as the table column name
%       Path        hierarchical channel path (e.g. "affect/depicted/emonet")
%       Class       top-level class (visual/audio/language/social/situation/affect)
%       Subclass    second level (e.g. "high_level_static"), or "(direct)"
%       Leaf        channel leaf name (e.g. "emonet")
%       Component   component label for a vector element (e.g. "Fear"), else ""
%       CompIndex   1-based index within the channel (1 for scalars)
%       Model       model/tool that produced it
%       Dtype       scalar | bool | event | vector | categorical | label
%       Level       "low" | "mid" | "high"  (perceptual-abstraction level)
%       Numeric     true if it enters the numeric matrix (scalar/bool/event/vector)
%       IsEmbedding true for the opaque SigLIP/DINOv2/CLAP embedding dimensions
%       Color       1x3 RGB for the class (for color-coded plots)
%
%   Only Numeric==true rows appear in the numeric matrix X; categorical "*_top"
%   codes and free-text label channels (asr_text, scene_description, ...) are listed
%   here for reference but excluded from X (their codes are not magnitudes).
%
%   See also FEATURESTOTABLE, READANNOTATIONCORPUSFULL, PLOTFEATUREMATRIX.

arguments
    templatePath (1,1) string = fullfile(fileparts(fileparts(mfilename("fullpath"))), ...
                                          "schema", "channel_template.json")
end

t = jsondecode(fileread(templatePath));
chans = t.channels;
if ~iscell(chans), chans = num2cell(chans); end       % normalize to cell of structs

Path=strings(0,1); Class=strings(0,1); Subclass=strings(0,1); Leaf=strings(0,1);
Component=strings(0,1); CompIndex=zeros(0,1); Model=strings(0,1); Dtype=strings(0,1);
Level=strings(0,1); Numeric=false(0,1); IsEmbedding=false(0,1);

for i = 1:numel(chans)
    c = chans{i};
    p = string(c.path);
    parts = split(p, "/");
    cls  = parts(1);
    sub  = "(direct)"; if numel(parts) > 2, sub = parts(2); end
    leaf = parts(end);
    dt   = string(c.dtype);
    isNum = ismember(dt, ["scalar", "bool", "event", "vector"]);
    isEmb = ismember(leaf, ["siglip_embedding", "dino_embedding", "clap_embedding"]);
    lev   = localLevel(cls, sub, leaf);
    mdl   = ""; if isfield(c, "model") && ~isempty(c.model), mdl = string(c.model); end

    if dt == "vector"
        comps = c.components;
        D = double(c.dim);
        if isempty(comps), comps = arrayfun(@(k) "d"+string(k-1), (1:D)'); end
        for k = 1:D
            comp = string(comps{k});
            Path(end+1,1)=p; Class(end+1,1)=cls; Subclass(end+1,1)=sub; Leaf(end+1,1)=leaf; %#ok<*AGROW>
            Component(end+1,1)=comp; CompIndex(end+1,1)=k; Model(end+1,1)=mdl;
            Dtype(end+1,1)=dt; Level(end+1,1)=lev; Numeric(end+1,1)=isNum; IsEmbedding(end+1,1)=isEmb;
        end
    else
        Path(end+1,1)=p; Class(end+1,1)=cls; Subclass(end+1,1)=sub; Leaf(end+1,1)=leaf;
        Component(end+1,1)=""; CompIndex(end+1,1)=1; Model(end+1,1)=mdl;
        Dtype(end+1,1)=dt; Level(end+1,1)=lev; Numeric(end+1,1)=isNum; IsEmbedding(end+1,1)=isEmb;
    end
end

info = table(Path, Class, Subclass, Leaf, Component, CompIndex, Model, Dtype, ...
             Level, Numeric, IsEmbedding);

% Stable, unique, valid column names: base = path with "/"->"__"; components get a
% "__<comp>" (or "__d<k-1>" for embeddings) suffix.
base = replace(info.Path, "/", "__");
suffix = strings(height(info), 1);
isVec = info.Dtype == "vector";
suffix(isVec & info.IsEmbedding) = "__d" + string(info.CompIndex(isVec & info.IsEmbedding) - 1);
mask = isVec & ~info.IsEmbedding;
suffix(mask) = "__" + info.Component(mask);
raw = base + suffix;
vn = matlab.lang.makeValidName(raw);
vn = matlab.lang.makeUniqueStrings(vn, {}, namelengthmax);
info.VarName = vn;

% Per-class color (matches analysis/figures/feature_map.svg).
info.Color = localColor(info.Class);

info = movevars(info, "VarName", "Before", "Path");
end

% -------------------------------------------------------------------------
function lev = localLevel(cls, sub, leaf)
% Perceptual-abstraction level per channel; mirrors the FEATURE_MAP summary table.
lev = "high";                                            % default (high-level semantics)
switch cls
    case "visual"
        if ismember(sub, ["low_level_static", "dynamic_motion"])
            lev = "low";
        elseif ismember(sub, ["saliency_aesthetics_depth", "faces_bodies_gaze"])
            lev = "mid";
        end                                              % else high (high_level_static, action)
    case "audio"
        if sub == "low_level"
            lev = "low";
        elseif leaf == "speech_present" || leaf == "word_rate"
            lev = "mid";
        end                                              % else high (high_level, voice_*)
    case "language"
        if ismember(leaf, ["freq_zipf", "word_length"])
            lev = "low";
        elseif sub == "syntax" || ismember(leaf, ["valence","arousal","dominance","concreteness","aoa"])
            lev = "mid";
        end                                              % else high (surprisal, entropy)
    case "social"
        if ismember(leaf, ["n_agents", "min_pair_distance"]), lev = "mid"; end
    case "affect"
        if ismember(leaf, ["face_emotion", "face_valence", "face_arousal"]), lev = "mid"; end
end
end

% -------------------------------------------------------------------------
function C = localColor(classCol)
map = ["visual", "#6366f1"; "audio", "#06b6d4"; "language", "#f59e0b"; ...
       "social", "#ec4899"; "situation", "#10b981"; "affect", "#ef4444"];
C = zeros(numel(classCol), 3);
for i = 1:numel(classCol)
    j = find(map(:,1) == classCol(i), 1);
    hex = "#808080"; if ~isempty(j), hex = map(j,2); end
    C(i,:) = double([hex2dec(extractBetween(hex,2,3)), hex2dec(extractBetween(hex,4,5)), ...
                     hex2dec(extractBetween(hex,6,7))]) / 255;
end
end
