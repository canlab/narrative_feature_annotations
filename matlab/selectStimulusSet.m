function sel = selectStimulusSet(C, opts)
%SELECTSTIMULUSSET Experimental-design selection of high-variance, low-redundancy segments.
%   SEL = SELECTSTIMULUSSET(C) takes the corpus struct from READANNOTATIONCORPUS,
%   splits every stimulus into fixed-length candidate segments, and greedily selects
%   a subset that maximizes the generalized variance (log-determinant of the
%   covariance) of the concatenated annotation time series projected onto the leading
%   principal components. Maximizing log det(cov) simultaneously rewards HIGH VARIANCE
%   across the major feature dimensions and INDEPENDENCE (low cross-correlation) of
%   the feature time series -- i.e. a D-optimal stimulus set for the annotation space.
%
%   Returns SEL with fields:
%     .table     - table of selected segments (rank, stimulus, t_start, t_end, dur_s)
%     .objTrace  - log det(cov) after each greedy addition
%     .randTrace - mean log det(cov) for random selection (baseline)
%     .idx       - candidate indices in selection order
%     .Y, .seg   - PC scores and per-segment metadata
%     .explained - PCA variance explained (for reference)
%
%   Options:
%     "SegLength" candidate segment length in seconds      (default 10)
%     "NumPCs"    number of leading PCs to design over      (default 10)
%     "K"         number of segments to select             (default 20)
%     "MaxNaNFrac"/"Lambda"/"OutDir"/"Save"/"Plot"          (see code)
%
%   See also READANNOTATIONCORPUS, ANALYZECORPUS.

arguments
    C (1,1) struct
    opts.SegLength (1,1) double = 10
    opts.NumPCs (1,1) double = 10
    opts.K (1,1) double = 20
    opts.MaxNaNFrac (1,1) double = 0.40
    opts.Lambda (1,1) double = 1e-3
    opts.OutDir (1,1) string = "analysis/figures"
    opts.Save (1,1) logical = true
    opts.Plot (1,1) logical = true
end

% ---- leading PCs of the annotation set (NaN-aware z-score) ----------------
keep = mean(isnan(C.X), 1) <= opts.MaxNaNFrac;
X = C.X(:, keep);
imputed = isnan(X);                       % remember what was missing (see below)
mu = mean(X, 1, "omitnan");
sg = std(X, 0, 1, "omitnan"); sg(sg == 0 | isnan(sg)) = 1;
Xz = (X - mu) ./ sg; Xz(isnan(Xz)) = 0;
[~, score, ~, ~, explained] = pca(Xz);
q = min(opts.NumPCs, size(score, 2));
Y = score(:, 1:q);

% ---- candidate segments (per stimulus, fixed-length windows) --------------
% Segments dominated by imputed (missing->mean) cells would sit exactly at the
% grand mean and can win the log-det objective BECAUSE their data are missing;
% exclude candidates whose imputed fraction exceeds MaxNaNFrac.
seg = struct("stim", {}, "t0", {}, "t1", {}, "n", {}, "s", {}, "S", {});
nDropped = 0;
row = 1;
for i = 1:numel(C.ids)
    h = C.nT(i);
    rateHz = 1;
    try, rateHz = double(C.ann{i}.time.rate_hz); catch, end %#ok<CTCH>
    stepSamp = max(2, round(opts.SegLength * rateHz));   % SegLength is in SECONDS
    Yi = Y(row:row+h-1, :);
    Mi = imputed(row:row+h-1, :);
    ti = C.time_sec(row:row+h-1);
    row = row + h;
    edges = 0:stepSamp:h;
    for e = 1:numel(edges)-1
        idx = edges(e)+1 : min(edges(e+1), h);
        if numel(idx) < 2, continue; end
        if mean(Mi(idx, :), "all") > opts.MaxNaNFrac
            nDropped = nDropped + 1;
            continue
        end
        Yc = Yi(idx, :);
        seg(end+1) = struct("stim", C.ids(i), "t0", ti(idx(1)), "t1", ti(idx(end)), ...
            "n", numel(idx), "s", sum(Yc, 1).', "S", Yc.' * Yc); %#ok<AGROW>
    end
end
if nDropped > 0
    fprintf("  (%d candidate segments dropped: >%.0f%% imputed/missing cells)\n", ...
        nDropped, 100 * opts.MaxNaNFrac);
end
nSeg = numel(seg);
K = min(opts.K, nSeg);

% ---- greedy D-optimal selection -------------------------------------------
[idx, objTrace] = greedy(seg, K, q, opts.Lambda);

% ---- random baseline (mean log det over restarts) -------------------------
rng(0);                          % seed so the reported baseline is reproducible
randTrace = zeros(1, K);
nrep = 20;
for r = 1:nrep
    perm = randperm(nSeg, K);
    randTrace = randTrace + cumObj(seg(perm), q, opts.Lambda);
end
randTrace = randTrace / nrep;

% ---- assemble result ------------------------------------------------------
rank = (1:K).';
stimulus = arrayfun(@(j) seg(idx(j)).stim, 1:K).';
t_start = arrayfun(@(j) seg(idx(j)).t0, 1:K).';
t_end = arrayfun(@(j) seg(idx(j)).t1, 1:K).';
dur_s = t_end - t_start;
sel.table = table(rank, stimulus, t_start, t_end, dur_s);
sel.objTrace = objTrace; sel.randTrace = randTrace; sel.idx = idx;
sel.Y = Y; sel.seg = seg; sel.explained = explained;

fprintf("Selected %d of %d candidate %gs segments (%.1f min) from %d stimuli.\n", ...
    K, nSeg, opts.SegLength, sum(dur_s)/60, numel(unique(stimulus)));
fprintf("  log det(cov): greedy %.2f vs random %.2f (higher = more variance + independence).\n", ...
    objTrace(end), randTrace(end));

% ---- figure ---------------------------------------------------------------
if opts.Plot
    f = figure("Color", "w", "Position", [100 100 820 340]);
    tiledlayout(1, 2, "Padding", "compact");
    nexttile;
    plot(1:K, objTrace, "-o", "LineWidth", 1.5, "MarkerSize", 3); hold on;
    plot(1:K, randTrace, "--", "Color", [.6 .6 .6], "LineWidth", 1.2);
    xlabel("# segments selected"); ylabel("log det(cov)");
    legend(["greedy D-optimal", "random (mean)"], "Location", "southeast");
    title("Design objective: variance + independence"); grid on;
    nexttile;
    gscatter(Y(:, 1), Y(:, 2), [], [.8 .8 .8], ".", 4); hold on;
    for j = 1:K
        s = seg(idx(j));
        plot(s.s(1)/s.n, s.s(2)/s.n, "ro", "MarkerFaceColor", "r", "MarkerSize", 5);
    end
    legend off; xlabel("PC1"); ylabel("PC2");
    title("Selected segments (red) in PC space");
    if opts.Save
        if ~isfolder(opts.OutDir), mkdir(opts.OutDir); end
        exportgraphics(f, fullfile(opts.OutDir, "design_selection.png"), "Resolution", 130);
    end
end
end

% ======================= helpers ======================================
function [idx, objTrace] = greedy(seg, K, q, lambda)
nSeg = numel(seg);
avail = true(1, nSeg);
idx = zeros(1, K); objTrace = zeros(1, K);
n = 0; s = zeros(q, 1); S = zeros(q);
for step = 1:K
    bestObj = -inf; bestI = 0;
    for i = find(avail)
        o = objAdd(n, s, S, seg(i), q, lambda);
        if o > bestObj, bestObj = o; bestI = i; end
    end
    if bestI == 0, break; end
    avail(bestI) = false; idx(step) = bestI; objTrace(step) = bestObj;
    n = n + seg(bestI).n; s = s + seg(bestI).s; S = S + seg(bestI).S;
end
end

function o = objAdd(n, s, S, sg, q, lambda)
n2 = n + sg.n; s2 = s + sg.s; S2 = S + sg.S;
cov2 = (S2 - (s2 * s2.') / n2) / max(n2 - 1, 1) + lambda * eye(q);
o = logdet(cov2);
end

function tr = cumObj(segs, q, lambda)
n = 0; s = zeros(q, 1); S = zeros(q); tr = zeros(1, numel(segs));
for k = 1:numel(segs)
    n = n + segs(k).n; s = s + segs(k).s; S = S + segs(k).S;
    cov = (S - (s * s.') / n) / max(n - 1, 1) + lambda * eye(q);
    tr(k) = logdet(cov);
end
end

function v = logdet(A)
[L, p] = chol(A);
if p == 0, v = 2 * sum(log(diag(L))); else, v = -inf; end
end
