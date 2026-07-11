function stats = refreshAnalysis(corpusFolder, opts)
%REFRESHANALYSIS Re-run all corpus analyses and export the numbers the docs depend on.
%   STATS = REFRESHANALYSIS(CORPUSFOLDER) regenerates every analysis artifact after the
%   corpus changes, matching what the documentation cites:
%     - FULL corpus: summary stats + the design-selection figure (selectStimulusSet)
%     - AUDIOVISUAL subset: the structural figures (correlation heatmap, PCA scree,
%       feature/class networks) and class couplings — REVIEW_PAPER Figs 1-5 are captioned
%       "(audiovisual subset)" because affect/social/visual channels are undefined for
%       audio/text stimuli and drop out of full-corpus scalar analyses
%     - speech-rich vs speech-sparse class-coupling comparison (AV subset)
%   Writes analysis/corpus_stats.json with both full-corpus and AV-subset numbers.
%   Missing classes export as NaN (never silently empty).
%
%   Run after annotating new stimuli (Python side: tools/refresh_corpus.sh).
%
%   See also READANNOTATIONCORPUS, ANALYZECORPUS, SELECTSTIMULUSSET.

arguments
    corpusFolder (1,1) string = "annotations/corpus"
    opts.OutDir (1,1) string = "analysis/figures"
    opts.StatsFile (1,1) string = "analysis/corpus_stats.json"
end

C = readAnnotationCorpus(corpusFolder);

% ---- per-stimulus rate + modality (rate-aware durations; AV subset) --------
N = numel(C.ids);
rates = ones(1, N);
mods = strings(1, N);
for i = 1:N
    try, rates(i) = double(C.ann{i}.time.rate_hz); catch, end %#ok<CTCH>
    mods(i) = string(C.ann{i}.stimulus.modality);
end
totalMin = sum(C.nT ./ rates) / 60;

% ---- FULL corpus: stats + design tool (figure saved here) ------------------
resFull = analyzeCorpus(C, "OutDir", opts.OutDir, "Save", false);
sel = selectStimulusSet(C, "OutDir", opts.OutDir);        % saves design_selection.png

% ---- AUDIOVISUAL subset: structural figures + class couplings --------------
isAV = mods == "audiovisual";
if any(isAV)
    rowsAV = ismember(C.stim, C.ids(isAV));
    Cav = C;
    Cav.ids = C.ids(isAV); Cav.ann = C.ann(isAV); Cav.nT = C.nT(isAV);
    Cav.X = C.X(rowsAV, :); Cav.stim = C.stim(rowsAV); Cav.time_sec = C.time_sec(rowsAV);
    resAV = analyzeCorpus(Cav, "OutDir", opts.OutDir, "Save", true);  % Figs 1-4 (AV)
    grp = groupContingency(Cav, opts.OutDir);                          % Fig 5 (AV)
else
    resAV = resFull;
    grp = groupContingency(C, opts.OutDir);
end

% ---- assemble stats ---------------------------------------------------------
[srcs, ~, si] = unique(arrayfun(@(i) string(C.ann{i}.stimulus.source), 1:N), "stable");
bySource = struct();
for k = 1:numel(srcs), bySource.(matlab.lang.makeValidName(srcs(k))) = sum(si == k); end

stats = struct( ...
    "n_stimuli", N, ...
    "total_minutes", round(totalMin, 1), ...
    "n_timepoints", size(C.X, 1), ...
    "n_channels_total", numel(C.channels), ...
    "by_source", bySource, ...
    "by_modality", struct("audiovisual", sum(mods == "audiovisual"), ...
                          "audio_only", sum(mods == "audio-only"), ...
                          "text_only", sum(mods == "text-only")), ...
    "full", pcaStats(resFull), ...
    "av_subset", setfield(pcaStats(resAV), "n_stimuli", sum(isAV)), ... %#ok<SFLD>
    "class_visual_social", classCoupling(resAV, "visual", "social"), ...
    "class_audio_social", classCoupling(resAV, "audio", "social"), ...
    "class_audio_visual", classCoupling(resAV, "audio", "visual"), ...
    "class_affect_social", classCoupling(resAV, "affect", "social"), ...
    "class_affect_visual", classCoupling(resAV, "affect", "visual"), ...
    "class_affect_situation", classCoupling(resAV, "affect", "situation"), ...
    "class_affect_audio", classCoupling(resAV, "affect", "audio"), ...
    "design_greedy", round(sel.objTrace(end), 1), ...
    "design_random", round(sel.randTrace(end), 1), ...
    "design_k", height(sel.table), ...
    "design_n_source_stimuli", numel(unique(sel.table.stimulus)), ...
    "contingency", grp);

fid = fopen(opts.StatsFile, "w");
if fid == -1
    error("refreshAnalysis:statsFile", "Cannot open %s for writing.", opts.StatsFile);
end
fprintf(fid, "%s", jsonencode(stats, "PrettyPrint", true));
fclose(fid);
fprintf("Wrote %s. Corpus: %d stimuli, %.1f min. Full: %d PCs to 80%%; AV subset: %d.\n", ...
    opts.StatsFile, N, totalMin, stats.full.pcs_to_80, stats.av_subset.pcs_to_80);
end

% =========================================================================
function s = pcaStats(res)
s = struct("n_channels_analyzed", numel(res.channels), ...
    "pcs_to_80", find(cumsum(res.explained) >= 80, 1), ...
    "pcs_to_90", find(cumsum(res.explained) >= 90, 1), ...
    "pc1_5_pct", round(sum(res.explained(1:min(5, end))), 1));
end

function v = classCoupling(res, a, b)
% Mean |r| between two feature classes; NaN (never empty) if either is absent.
ia = res.classes == a; ib = res.classes == b;
if any(ia) && any(ib), v = round(res.classR(ia, ib), 2); else, v = NaN; end
end

% -------------------------------------------------------------------------
function g = groupContingency(C, outdir)
% Split stimuli by speech density (median per-stimulus mean word rate) and compare
% the feature-class coupling matrix between the two halves. NaN-safe: returns a
% NaN-filled struct when the split is impossible instead of crashing.
g = struct("n_speech_rich", NaN, "n_speech_sparse", NaN, ...
    "visual_social_rich", NaN, "visual_social_sparse", NaN, ...
    "audio_social_rich", NaN, "audio_social_sparse", NaN);
ci = C.channels == "audio__speech__word_rate";
if ~any(ci), warning("refreshAnalysis:noWordRate", "word_rate missing; skipping contingency."); return; end
mu = zeros(1, numel(C.ids));
for i = 1:numel(C.ids), mu(i) = mean(C.X(C.stim == C.ids(i), ci), "omitnan"); end
mu(isnan(mu)) = 0;
if numel(unique(mu)) < 2, warning("refreshAnalysis:constantWordRate", "word_rate constant; skipping contingency."); return; end
grp = {C.ids(mu > median(mu)), C.ids(mu <= median(mu))};
gname = ["speech-rich", "speech-sparse"];

keep = mean(isnan(C.X), 1) <= 0.40;
X = C.X(:, keep);
cls = extractBefore(C.channels(keep), "__");
[classes, ~, cidx] = unique(cls, "stable");
nc = numel(classes);
M = cell(1, 2);
for gi = 1:2
    rows = ismember(C.stim, grp{gi});
    if sum(rows) < 2, return; end
    R = corr(X(rows, :), "rows", "pairwise"); R(isnan(R)) = 0;
    Mg = nan(nc);
    for a = 1:nc
        for b = 1:nc
            blk = abs(R(cidx == a, cidx == b));
            if a == b, blk(logical(eye(sum(cidx == a)))) = NaN; end
            Mg(a, b) = mean(blk(:), "omitnan");
        end
    end
    M{gi} = Mg;
end

look = @(Mg, a, b) lookupPair(Mg, classes, a, b);
g.n_speech_rich = numel(grp{1});
g.n_speech_sparse = numel(grp{2});
g.visual_social_rich = look(M{1}, "visual", "social");
g.visual_social_sparse = look(M{2}, "visual", "social");
g.audio_social_rich = look(M{1}, "audio", "social");
g.audio_social_sparse = look(M{2}, "audio", "social");

f = figure("Color", "w", "Position", [100 100 760 340]);
tiledlayout(1, 2, "Padding", "compact");
for gi = 1:2
    nexttile; imagesc(M{gi}, [0 .3]); axis square; colorbar;
    set(gca, "XTick", 1:nc, "XTickLabel", classes, "YTick", 1:nc, "YTickLabel", classes, ...
        "TickLabelInterpreter", "none"); xtickangle(45); title(gname(gi));
end
colormap(parula);
if ~isfolder(outdir), mkdir(outdir); end
exportgraphics(f, fullfile(outdir, "class_coupling_by_group.png"), "Resolution", 130);
end

function v = lookupPair(Mg, classes, a, b)
ia = classes == a; ib = classes == b;
if any(ia) && any(ib), v = round(Mg(ia, ib), 2); else, v = NaN; end
end
