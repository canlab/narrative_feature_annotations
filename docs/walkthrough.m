%% Narrative Feature Extraction — MATLAB walkthrough
% Run this section by section (Ctrl+Enter / "Run Section"). It tours the common
% operations: loading an annotation, inspecting channels, watching a movie with its
% annotations synced underneath, and analyzing / designing across the whole corpus.
% See docs/CONTENTS.md for the full guide.

%% 0. Setup  — put matlab/ on the path and cd to the project root
clear here proj
here = pwd;
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
% C holds ALL clips concatenated, NOT a single movie/story. Structure of C:
%   C.X        [totalTimepoints x channels] the stacked feature matrix. Here 28237 x 65.
%              ROWS are timepoints (1 per second of the common grid) from every clip
%              stacked end-to-end -- they are NOT seconds. Use C.time_sec for the
%              within-clip time and C.stim to know which clip a row belongs to.
%   C.stim     [totalTimepoints x 1] categorical stimulus id for each row of C.X
%   C.time_sec [totalTimepoints x 1] within-clip time (s) for each row
%   C.channels [1 x 65] the column names of C.X
%   C.ids      [1 x 83] stimulus ids ;  C.nT samples per clip ;  C.ann full structs
% IMPORTANT: readAnnotationCorpus returns ONLY the 65 scalar channels (luminance, RMS,
% word_rate, valence, ...). The multivariate model outputs -- the SigLIP / DINOv2 / CLAP
% embeddings, AudioSet tags, VideoMAE action posteriors, EmoNet, MFCC, etc. -- are NOT
% in this C.X. To get every variable (all ~2768, embeddings included) use the FULL
% reader in section 8 (readAnnotationCorpusFull), whose F.info gives, per column, the
% class/subclass/level/model and whether it is an embedding (F.info.IsEmbedding).
C = readAnnotationCorpus("annotations/corpus");
fprintf("corpus: %d stimuli, %d scalar channels, %d timepoints\n", ...
        numel(C.ids), numel(C.channels), size(C.X,1));
% Rows for one clip: pick them with the stimulus id, e.g.
%   r = C.stim == "BigBuckBunny";  plot(C.time_sec(r), C.X(r, 1));

%% 5.2 Read the FULL corpus

C = readAnnotationCorpusFull("annotations/corpus");
fprintf("corpus: %d clips, %d scalar channels, %d timepoints\n", ...
    numel(C.ids), size(C.X, 2), size(C.X, 1));

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

%% 8. FULL feature set: all ~2,768 variables, color-coded, + factor analysis
% readAnnotationCorpus (above) keeps only scalar channels. To get EVERY variable —
% each vector channel (SigLIP/DINOv2/CLAP embeddings, AudioSet/action posteriors,
% EmoNet, MFCC, ...) expanded into one column per component — use the FULL reader.
F = readAnnotationCorpusFull("annotations/corpus");    % F.X [timepoints x ~2768]
% F.info is the label table: one row per column, with class/subclass/level/model/color.
fprintf("full set: %d variables (%d embedding, %d interpretable) over %d timepoints\n", ...
        size(F.X,2), sum(F.info.IsEmbedding), sum(~F.info.IsEmbedding), size(F.X,1));

% Visualize the whole feature matrix for one clip, color-coded by category:
plotFeatureMatrix(F, "Clip", "BigBuckBunny");          % heatmap; class color strip on the left
% plotFeatureMatrix(F, "Mode","classmean");            % or six class-mean trajectories

% Exploratory factor analysis (interpretable features by default; excludes the opaque
% embeddings and the big AudioSet/Kinetics taxonomies). Returns rotated loadings +
% a per-timepoint factor-score time series you can plot like any feature.
fa = factorAnalysisCorpus(F, "NumFactors", 10);        % draws loadings heatmap + scree
r  = F.stim == "BigBuckBunny";
figure; plot(F.time_sec(r), fa.scores(r, 1:3)); legend("F1","F2","F3");
xlabel("time (s)"); ylabel("factor score"); title("Factor time series — BigBuckBunny");

%% 9. PER-CATEGORY factors: reduce each model's output to a few factors
% Runs factor analysis SEPARATELY within each category -- each of the 13 multivariate
% model outputs (SigLIP/DINOv2/CLAP embeddings, AudioSet, action, EmoNet, text-emotion,
% probes, MFCC, chroma, facial affect, text sentiment) AND the interpretable-scalar
% block -- and attaches the factor time series to F as F.extracted_factors. FACTORAN is
% used where a block's degrees of freedom allow; the high-dimensional opaque embeddings
% fall back to PCA. The .mat is saved for reuse (regenerable; kept in Dropbox only).
F = extractCategoryFactors(F, "NumFactors", 10, "Save", "analysis/extracted_factors.mat");
EF = F.extracted_factors;
fprintf("extracted %d factors across %d categories\n", size(EF.scores,2), numel(EF.byCategory));
disp(EF.labels(1:6, :))                                % FactorName, Category, Model, Class, Method, VarExpl

% Access one model's factors as a time series (e.g. the SigLIP visual embedding):
s  = EF.byCategory;
si = s(strcmp(string({s.name}), "siglip_embedding"));  % 768 dims -> 10 PCA factors
r  = F.stim == "BigBuckBunny";
figure; plot(F.time_sec(r), si.scores(r, 1:3));
legend("F1","F2","F3"); xlabel("time (s)"); ylabel("factor score");
title("SigLIP embedding factors — BigBuckBunny");

% Reload later without recomputing, and merge back into a corpus struct:
%   S = load("analysis/extracted_factors.mat");     % -> S.extracted_factors
%   F = readAnnotationCorpusFull("annotations/corpus");
%   F.extracted_factors = S.extracted_factors;      % rows align with F.X / F.stim

%% 10. Where to go next
% - Inspect any other stimulus: change `stimId` in section 0.
% - Annotate a NEW movie (Python):
%     PYTHONPATH=src .venv/bin/python -m nfe.run <movie> --vision --audio-hl --events \
%         --template schema/channel_template.json
% - SEARCH segments by feature in a browser (serve from the project root):
%     python3 tools/serve.py   % Range-enabled (video seeking works); open http://localhost:8000/analysis/web/index.html
% - Full reference: docs/CONTENTS.md ; format: docs/design/ANNOTATION_FORMAT.md
disp("Walkthrough complete. See docs/CONTENTS.md for the full guide.");
