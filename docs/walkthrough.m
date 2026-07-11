%% Narrative Feature Extraction — MATLAB walkthrough
% Run this section by section (Ctrl+Enter / "Run Section"). It tours the common
% operations: loading an annotation, inspecting channels, watching a movie with its
% annotations synced underneath, and analyzing / designing across the whole corpus.
% See docs/CONTENTS.md for the full guide.

%% 0. Setup  — put matlab/ on the path and cd to the project root
here = fileparts(mfilename("fullpath"));
if isempty(here); here = pwd; end       % "Run Section" mode: mfilename is empty
proj = fileparts(here);                 % docs/ -> project root
if ~isfolder(fullfile(proj, "matlab")) && isfolder(fullfile(here, "matlab"))
    proj = here;                        % already at the project root (common cwd)
end
assert(isfolder(fullfile(proj, "matlab")), ...
    "Run from the project root or docs/ (matlab/ not found from here).");
addpath(fullfile(proj, "matlab"));
cd(proj);
disp("Project root: " + string(proj));

% A stimulus we'll use throughout (has both a movie file and an annotation):
stimId  = "ses-01_run-01_order-04_content-parkour";
annDir  = fullfile("annotations", "corpus", stimId);
movie   = fullfile("data", "movies", "spacetop", "videos", "ses-01", stimId + ".mp4");

%% 1. Load and inspect ONE annotation
% readAnnotations accepts the stimulus folder, the .h5, or a JSON profile.
ann = readAnnotations(annDir);
disp(ann.stimulus)                       % id, modality, duration, source, ...
fprintf("grid: %d samples @ %g Hz\n", ann.time.n_samples, ann.time.rate_hz);

% The feature hierarchy: top-level classes and their subgroups
classes = string(fieldnames(ann.features))';
fprintf("feature classes: %s\n", strjoin(classes, ", "));

%% 2. Get a single feature channel by hierarchical path
% Scalars, vectors, labels, categoricals, and events are all retrievable.
lum = getFeature(ann, "visual/low_level_static/luminance");   % scalar [n]
mf  = getFeature(ann, "audio/low_level/mfcc");                 % vector [n x 13]
fprintf("luminance: %d samples, model=%s\n", numel(lum.value), lum.model);
fprintf("mfcc: [%d x %d], applicable=%d\n", size(mf.value,1), size(mf.value,2), mf.applicable);

% Channels carry provenance in their fields: model, version, native_rate_hz, resample.

%% 3. Scalar channels as a timetable, and a quick plot
tt = featuresToTimetable(ann);           % every scalar channel on the common grid
vars = ["visual__low_level_static__luminance", ...
        "audio__low_level__rms", ...
        "audio__speech__word_rate"];
figure("Color","w"); sp = stackedplot(tt(:, vars)); sp.Title = char(stimId);

%% 4. WATCH the movie with its annotation time series synced underneath  (interactive)
% Opens a window: video on top, time series below, a red marker tracking playback.
% Click "Play", or drag the slider to scrub. Close the window when done.
annotationMovieViewer(movie, annDir);
% Try custom channels / speed:
%   annotationMovieViewer(movie, annDir, "Channels", ...
%       ["audio/low_level/rms","visual/dynamic_motion/flow_magnitude"], "Speed", 1.5)

%% 5. Load the WHOLE corpus (all annotated stimuli) into one structure
C = readAnnotationCorpus("annotations/corpus");
% C.X is [totalTimepoints x channels]; C.stim / C.time_sec label each row.
fprintf("corpus: %d stimuli, %d channels, %d timepoints\n", ...
        numel(C.ids), numel(C.channels), size(C.X,1));

%% 6. Cross-feature STRUCTURE: correlation, PCA, network graphs
% Saves figures to analysis/figures/ and returns a results struct.
res = analyzeCorpus(C);
fprintf("PCs to reach 80%% variance: %d\n", find(cumsum(res.explained) >= 80, 1));
% Strongest feature-class couplings (mean |r| between classes):
M = res.classR; M(logical(eye(numel(res.classes)))) = NaN;
[mx, k] = max(M(:)); [a,b] = ind2sub(size(M), k);
fprintf("strongest class link: %s <-> %s (%.2f)\n", res.classes(a), res.classes(b), mx);

%% 7. EXPERIMENTAL DESIGN: select a high-variance, low-redundancy stimulus set
% Greedy D-optimal selection of segments that maximize log det(cov) of the
% concatenated annotation time series (variance across PCs + time-series independence).
sel = selectStimulusSet(C, "SegLength", 10, "NumPCs", 10, "K", 20);
disp(sel.table)                          % rank, stimulus, t_start, t_end, dur_s
% sel.objTrace vs sel.randTrace quantifies the gain over random selection.

%% 8. Where to go next
% - Inspect any other stimulus: change `stimId` in section 0.
% - Annotate a NEW movie (Python):
%     PYTHONPATH=src .venv/bin/python -m nfe.run <movie> --vision --audio-hl --events \
%         --template schema/channel_template.json
% - SEARCH segments by feature in a browser (serve from the project root):
%     python3 -m http.server 8000   % then open http://localhost:8000/analysis/web/index.html
% - Full reference: docs/CONTENTS.md ; format: docs/design/ANNOTATION_FORMAT.md
disp("Walkthrough complete. See docs/CONTENTS.md for the full guide.");
