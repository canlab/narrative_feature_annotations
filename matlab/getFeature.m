function ch = getFeature(ann, path)
%GETFEATURE Retrieve one feature channel by hierarchical path.
%   CH = GETFEATURE(ANN, 'visual/low_level_static/luminance') returns the
%   channel struct (with .value and metadata) at that path in ANN.features.
%   Errors if the path does not resolve to a channel.
%
%   See also READANNOTATIONS, FEATURESTOTIMETABLE.

arguments
    ann (1,1) struct
    path (1,1) string
end

parts = split(path, "/");
node = ann.features;
for i = 1:numel(parts)
    key = parts(i);
    if ~isfield(node, key)
        error("getFeature:notFound", "Path component '%s' not found.", key);
    end
    node = node.(key);
end
if ~(isstruct(node) && isfield(node, 'value'))
    error("getFeature:notAChannel", "Path '%s' does not resolve to a feature channel.", path);
end
ch = node;
end
