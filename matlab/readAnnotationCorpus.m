function C = readAnnotationCorpus(folder, opts)
%READANNOTATIONCORPUS Load a whole annotation corpus into an analysis structure.
%   C = READANNOTATIONCORPUS(FOLDER) reads every <id>/<id>.h5 under FOLDER (e.g.
%   "annotations/corpus") and returns a struct C with:
%       C.ids        - 1xN string array of stimulus ids
%       C.ann        - 1xN cell of full annotation structs (see READANNOTATIONS)
%       C.channels   - 1xP string array of scalar channel paths (shared set)
%       C.X          - [sumT x P] stacked scalar matrix (all timepoints, all stimuli)
%       C.stim       - [sumT x 1] categorical stimulus id per row of X
%       C.time_sec   - [sumT x 1] within-stimulus time for each row of X
%       C.nT         - 1xN samples per stimulus
%   The constant-shape contract (see ANNOTATION_FORMAT.md) guarantees every file
%   shares the same channel set, so X is rectangular and ready for correlation /
%   PCA / the Phase-4 design tool. Inapplicable channels are present as NaN.
%
%   Options:
%       "Pattern"     glob for files (default "*/*.h5")
%       "Class"       restrict scalar channels to one branch (e.g. "audio")
%       "DropAllNaN"  drop channels that are NaN for every stimulus (default true)
%
%   See also READANNOTATIONS, FEATURESTOTIMETABLE, ANNOTATIONMOVIEVIEWER.

arguments
    folder (1,1) string {mustBeFolder}
    opts.Pattern (1,1) string = "*/*.h5"
    opts.Class (1,1) string = ""
    opts.DropAllNaN (1,1) logical = true
end

files = dir(fullfile(folder, opts.Pattern));
if isempty(files)
    error("readAnnotationCorpus:none", "No annotation files matching %s under %s.", ...
        opts.Pattern, folder);
end

ids = strings(1, 0);
annCell = {};
tts = {};
for i = 1:numel(files)
    f = string(fullfile(files(i).folder, files(i).name));
    try
        ann = readAnnotations(f);                 % skip partially-written / corrupt files
    catch ME
        warning("readAnnotationCorpus:skip", "Skipping %s (%s).", files(i).name, ME.message);
        continue
    end
    ids(end+1) = string(ann.stimulus.id); %#ok<AGROW>
    annCell{end+1} = ann; %#ok<AGROW>
    tts{end+1} = featuresToTimetable(ann); %#ok<AGROW>
end
N = numel(ids);
if N == 0
    error("readAnnotationCorpus:none", "No readable annotation files under %s.", folder);
end
C.ids = ids;
C.ann = annCell;

% Shared scalar-channel set: intersection of variable names across stimuli (the
% constant-shape template makes these identical, but intersect is robust).
names = string(tts{1}.Properties.VariableNames);
for i = 2:N
    names = intersect(names, string(tts{i}.Properties.VariableNames), "stable");
end
if opts.Class ~= ""
    names = names(startsWith(names, opts.Class + "__"));
end

% Stack into one tall matrix, aligning columns by name.
nT = cellfun(@height, tts);
X = nan(sum(nT), numel(names));
stim = strings(sum(nT), 1);
tvec = nan(sum(nT), 1);
row = 1;
for i = 1:N
    h = nT(i);
    Ti = tts{i};
    sub = Ti{:, cellstr(names)};
    X(row:row+h-1, :) = sub;
    stim(row:row+h-1) = C.ids(i);
    tvec(row:row+h-1) = seconds(Ti.Properties.RowTimes);
    row = row + h;
end

if opts.DropAllNaN
    keep = ~all(isnan(X), 1);
    X = X(:, keep);
    names = names(keep);
end

C.channels = names;
C.X = X;
C.stim = categorical(stim);
C.time_sec = tvec;
C.nT = nT;
fprintf("Loaded %d stimuli, %d shared scalar channels, %d total timepoints.\n", ...
    N, numel(names), size(X, 1));
end
