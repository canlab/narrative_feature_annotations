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

[X, info] = featuresToMatrix(ann, info, "IncludeEmbeddings", opts.IncludeEmbeddings, ...
                             "NaNInapplicable", opts.NaNInapplicable);
rowTimes = seconds(ann.time_sec(:));
T = array2timetable(X, "RowTimes", rowTimes, "VariableNames", cellstr(info.VarName));
end
