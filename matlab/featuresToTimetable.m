function tt = featuresToTimetable(ann)
%FEATURESTOTIMETABLE Collect all scalar feature channels into a timetable.
%   TT = FEATURESTOTIMETABLE(ANN) walks ANN.features and gathers every leaf
%   channel whose value is a numeric column matching the common time grid,
%   one variable per channel, named by its hierarchical path (slashes -> __).
%   Non-scalar channels (vectors, labels) are skipped.
%
%   See also READANNOTATIONS, GETFEATURE.

arguments
    ann (1,1) struct
end

n = numel(ann.time_sec);
names = strings(0,1);
cols = {};
[names, cols] = collect(ann.features, "", n, names, cols);

rowTimes = seconds(ann.time_sec);
if isempty(cols)
    tt = timetable('Size', [n 0], 'RowTimes', rowTimes);
    return;
end
tt = timetable(rowTimes, cols{:}, 'VariableNames', cellstr(names));
end

% -------------------------------------------------------------------------
function [names, cols] = collect(g, prefix, n, names, cols)
fn = fieldnames(g);
for i = 1:numel(fn)
    node = g.(fn{i});
    if prefix == ""
        key = string(fn{i});
    else
        key = prefix + "__" + string(fn{i});
    end
    if isstruct(node) && isfield(node, 'value')
        v = node.value;
        % Continuous/numeric channels only. Exclude categorical class codes and
        % label/vector channels (e.g. audioset_top) — their integer codes are not
        % magnitudes and must not be z-scored/correlated as features.
        okType = ~isfield(node, 'dtype') || ...
                 ismember(string(node.dtype), ["scalar", "bool", "event"]);
        if okType && isnumeric(v) && isvector(v) && numel(v) == n
            v = double(v(:));
            % Not-applicable channels carry fill values (event 0s, bool -1s) that are
            % NOT measurements — represent them as NaN so they never enter analyses.
            if isfield(node, 'applicable') && ~node.applicable
                v(:) = NaN;
            end
            names(end+1,1) = key; %#ok<AGROW>
            cols{end+1} = v; %#ok<AGROW>
        end
    elseif isstruct(node)
        [names, cols] = collect(node, key, n, names, cols);
    end
end
end
