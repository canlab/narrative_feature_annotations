function C = readAnnotationCorpusFull(folder, opts)
%READANNOTATIONCORPUSFULL Load the whole corpus as ONE wide feature matrix.
%   C = READANNOTATIONCORPUSFULL(FOLDER) reads every <id>/<id>.h5 under FOLDER,
%   expands every channel into its component variables (FEATURESTOTABLE), and
%   vertically concatenates all clips into a single table/matrix. Unlike
%   READANNOTATIONCORPUS (scalar channels only), this returns the full expanded
%   ~2.7k-variable set, ready for time-series visualization and factor analysis.
%
%   Returns struct C with:
%       C.info     [V x k] FEATUREINFO label table (row i describes column i of X)
%       C.X        [sumT x V] stacked numeric matrix (all timepoints, all clips)
%       C.T        table: [Stim, TimeSec, <V feature columns>] (X with keys)
%       C.stim     [sumT x 1] categorical stimulus id per row
%       C.time_sec [sumT x 1] within-stimulus time (s) per row
%       C.ids      1xN string ids;  C.nT 1xN samples per clip
%
%   Name-value options:
%       "Pattern"           ("*/*.h5") glob under FOLDER
%       "IncludeEmbeddings" (true)  keep opaque SigLIP/DINOv2/CLAP dims
%       "DropAllNaN"        (true)  drop variables that are NaN for every clip
%       "Verbose"           (true)  print progress
%
%   Example:
%       C = readAnnotationCorpusFull("annotations/corpus");
%       plotFeatureMatrix(C);                 % color-coded by category
%       F = factorAnalysisCorpus(C, "NumFactors", 8);
%
%   See also FEATURESTOTABLE, FEATUREINFO, PLOTFEATUREMATRIX, FACTORANALYSISCORPUS.

arguments
    folder (1,1) string {mustBeFolder}
    opts.Pattern (1,1) string = "*/*.h5"
    opts.IncludeEmbeddings (1,1) logical = true
    opts.DropAllNaN (1,1) logical = true
    opts.BuildTable (1,1) logical = false     % also build C.T (a wide table); slow for many vars
    opts.Verbose (1,1) logical = true
end

files = dir(fullfile(folder, opts.Pattern));
if isempty(files)
    error("readAnnotationCorpusFull:none", "No files matching %s under %s.", opts.Pattern, folder);
end

info = featureInfo();
Xcell = {}; ids = strings(1,0); nT = []; stimCell = {}; timeCell = {};
for i = 1:numel(files)
    f = string(fullfile(files(i).folder, files(i).name));
    try
        ann = readAnnotations(f);
        % Build the numeric matrix directly (NOT a per-clip wide table — table ops
        % scale terribly with column count and would take orders of magnitude longer
        % on the ~7.9k-variable set).
        Xi = featuresToMatrix(ann, info, "IncludeEmbeddings", opts.IncludeEmbeddings);
    catch ME
        warning("readAnnotationCorpusFull:skip", "Skipping %s (%s).", files(i).name, ME.message);
        continue
    end
    id = string(ann.stimulus.id);
    ni = size(Xi, 1);
    ids(end+1) = id; %#ok<AGROW>
    nT(end+1) = ni; %#ok<AGROW>
    Xcell{end+1} = Xi; %#ok<AGROW>
    stimCell{end+1} = repmat(id, ni, 1); %#ok<AGROW>
    timeCell{end+1} = ann.time_sec(:); %#ok<AGROW>
    if opts.Verbose, fprintf("  [%2d/%2d] %-42s %5d x %d\n", i, numel(files), id, ni, size(Xi,2)); end
end
if isempty(ids), error("readAnnotationCorpusFull:none", "No readable files under %s.", folder); end

keepInfo = info(info.Numeric, :);
if ~opts.IncludeEmbeddings, keepInfo = keepInfo(~keepInfo.IsEmbedding, :); end

X = vertcat(Xcell{:});
stim = categorical(vertcat(stimCell{:}));
time_sec = vertcat(timeCell{:});

if opts.DropAllNaN
    good = ~all(isnan(X), 1);
    X = X(:, good);
    keepInfo = keepInfo(good, :);
    if opts.Verbose
        fprintf("Dropped %d all-NaN variables; %d remain.\n", sum(~good), sum(good));
    end
end

C.info = keepInfo;
C.X = X;
C.stim = stim;
C.time_sec = time_sec;
C.ids = ids;
C.nT = nT;
% The wide table (Stim + TimeSec + one column per variable) is convenient but
% array2table with thousands of variables is very slow, so it is opt-in.
if opts.BuildTable
    C.T = [table(stim, time_sec, 'VariableNames', {'Stim','TimeSec'}), ...
           array2table(X, 'VariableNames', cellstr(keepInfo.VarName))];
end

if opts.Verbose
    fprintf("Corpus: %d clips, %d timepoints, %d variables (%d embedding, %d interpretable).\n", ...
        numel(ids), size(X,1), size(X,2), sum(keepInfo.IsEmbedding), sum(~keepInfo.IsEmbedding));
end
end
