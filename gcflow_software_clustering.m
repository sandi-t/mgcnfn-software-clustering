function gcflow_software_clustering()
clc; close all;

cfg = struct();

cfg.adj_files = {
    './javacc/adjacency_matrix_aggregation.txt', ...
    './javacc/adjacency_matrix_association.txt', ...
    './javacc/adjacency_matrix_composition.txt', ...
    './javacc/adjacency_matrix_dependency.txt'
};
cfg.X_path      = './javac/node_features.txt';
cfg.labels_path = './javacc/label.txt';

cfg.num_classes = 4;
cfg.num_packages = 4;

cfg.k_selection = 'silhouette';

cfg.seed_start = 1;
cfg.seed_end   = 3;

cfg.gcn_hidden  = 128;
cfg.latent_dim  = 64;
cfg.flow.numBlocks = 4;
cfg.flow.hidden    = 64;

cfg.gmm.mean_scale_init = 3.0;
cfg.gmm.cov_scale_init  = 1.0;

cfg.maxEpochs   = 300;
cfg.lr          = 1e-3;
cfg.grad_clip   = 50;
cfg.patience    = 30;
cfg.lambda_unlabeled = 0.1;
cfg.weight_decay = 5e-4;

cfg.kmeans_reps = 20;
cfg.kmeans_iter = 500;
cfg.k_min = 3;
cfg.k_max = 20;

cfg.outdir = fullfile(pwd, 'gcflow_experiments');
if ~exist(cfg.outdir, 'dir'), mkdir(cfg.outdir); end
cfg.plot_save = true;
cfg.verbose   = true;

fprintf('=== GC-Flow Baseline for Software Module Clustering ===\n');
fprintf('Output directory: %s\n\n', cfg.outdir);

if exist(cfg.X_path, 'file')
    X = double(load(cfg.X_path));
else
    error('Feature file not found: %s', cfg.X_path);
end
N = size(X, 1);
D_in = size(X, 2);

V = numel(cfg.adj_files);
A_union = zeros(N, N);
for v = 1:V
    if ~exist(cfg.adj_files{v}, 'file')
        error('Adjacency file not found: %s', cfg.adj_files{v});
    end
    A = double(load(cfg.adj_files{v}));
    A = (A + A') / 2;
    A_union = A_union + A;
end
A = double(A_union > 0);
fprintf('Union adjacency built from %d relation graphs.\n', V);

has_gt = false;
gt_labels = [];
if exist(cfg.labels_path, 'file')
    gt_labels = load(cfg.labels_path);
    gt_labels = gt_labels(:);
    if length(gt_labels) == N
        has_gt = true;
        fprintf('Ground truth labels loaded from %s\n', cfg.labels_path);
    else
        warning('Label count mismatch, ignoring ground truth.');
    end
end

A_tilde = A + eye(N);
deg = sum(A_tilde, 2);
Dm = diag(1 ./ sqrt(deg + eps));
A_norm = Dm * A_tilde * Dm;

log_det_A = compute_log_abs_det(A_norm);
if cfg.verbose
    fprintf('log|det(A_norm)| = %.4f\n', log_det_A);
end

if D_in > 200
    fprintf('Reducing feature dimension from %d to 100 using PCA\n', D_in);
    [coeff, ~, ~] = pca(X);
    X = X * coeff(:, 1:100);
    D_in = size(X, 2);
end
X = X ./ (sqrt(sum(X.^2, 2)) + eps);

labeled_mask = false(N,1);
labels_onehot = [];
if has_gt
    labeled_mask = (gt_labels > 0);
    if sum(labeled_mask) > 0
        [classes, ~, class_idx] = unique(gt_labels(labeled_mask));
        C = length(classes);
        n_labeled = sum(labeled_mask);
        labels_onehot = zeros(n_labeled, C);
        labeled_indices = find(labeled_mask);
        for i = 1:n_labeled
            labels_onehot(i, class_idx(i)) = 1;
        end
        cfg.num_classes = C;
        cfg.num_packages = C;
        fprintf('Labeled nodes: %d / %d, detected %d classes\n', sum(labeled_mask), N, C);
    else
        warning('No labeled nodes found, falling back to unsupervised training.');
        labeled_mask(:) = false;
    end
end

seeds = cfg.seed_start : cfg.seed_end;
nSeeds = numel(seeds);

results = zeros(nSeeds, 12);
result_labels = cell(nSeeds, 1);
loss_curves = cell(nSeeds, 1);

for si = 1:nSeeds
    seed = seeds(si);
    rng(seed, 'twister');
    
    K = cfg.num_classes;
    d = cfg.latent_dim;
    init_mean_scale = cfg.gmm.mean_scale_init;
    init_cov_scale = cfg.gmm.cov_scale_init;
    init_log_weights = log(ones(1,K)/K);
    
    if cfg.verbose
        fprintf('\n===== Seed %d (%d/%d) =====\n', seed, si, nSeeds);
    end

    W1 = glorot_init(d, D_in);
    b1 = zeros(1, d);
    W2 = glorot_init(d, d);
    b2 = zeros(1, d);
    flow = cell(cfg.flow.numBlocks, 1);
    for b = 1:cfg.flow.numBlocks
        blk = struct();
        if mod(b,2) == 1
            blk.mask = [ones(1, ceil(d/2)), zeros(1, floor(d/2))];
        else
            blk.mask = [zeros(1, ceil(d/2)), ones(1, floor(d/2))];
        end
        m = sum(blk.mask == 1);
        n_out = sum(blk.mask == 0);
        blk.s.W1 = glorot_init(cfg.flow.hidden, m);
        blk.s.b1 = zeros(1, cfg.flow.hidden);
        blk.s.W2 = glorot_init(n_out, cfg.flow.hidden);
        blk.s.b2 = zeros(1, n_out);
        blk.t.W1 = glorot_init(cfg.flow.hidden, m);
        blk.t.b1 = zeros(1, cfg.flow.hidden);
        blk.t.W2 = glorot_init(n_out, cfg.flow.hidden);
        blk.t.b2 = zeros(1, n_out);
        flow{b} = blk;
    end

    gmm_mean_scale = dlarray(init_mean_scale * ones(1,K));
    gmm_cov_scale  = dlarray(init_cov_scale * ones(1,K));
    gmm_log_weights = dlarray(init_log_weights);

    params = pack_params_dl(W1, b1, W2, b2, flow, cfg.flow.numBlocks, ...
                            gmm_mean_scale, gmm_cov_scale, gmm_log_weights);
    adam = init_adam_state(params);
    best_loss = inf;
    patience_cnt = 0;
    loss_history = zeros(cfg.maxEpochs, 1);
    best_U = [];

    X_dl = dlarray(X);
    A_dl = dlarray(A_norm);
    labels_onehot_dl = [];
    if ~isempty(labels_onehot)
        labels_onehot_dl = dlarray(labels_onehot);
    end

    for epoch = 1:cfg.maxEpochs
        [loss_val, grads, fwd] = dlfeval(@(p) model_gradients(...
            p, X_dl, A_dl, labels_onehot_dl, labeled_mask, ...
            cfg, log_det_A, K, d, cfg.flow.numBlocks), params);
        lv = double(gather(extractdata(loss_val)));
        loss_history(epoch) = lv;

        grads = clip_gradients(grads, cfg.grad_clip);
        [params, adam] = adam_step(params, grads, adam, cfg.lr, cfg.weight_decay);

        if cfg.verbose && (mod(epoch, 50)==0 || epoch==1)
            fprintf('  Epoch %4d | loss = %.5f\n', epoch, lv);
        end

        if lv < best_loss - 1e-6
            best_loss = lv;
            patience_cnt = 0;
            best_U = double(gather(extractdata(fwd.U)));
        else
            patience_cnt = patience_cnt + 1;
            if patience_cnt >= cfg.patience
                if cfg.verbose
                    fprintf('  Early stopping at epoch %d\n', epoch);
                end
                loss_history = loss_history(1:epoch);
                break;
            end
        end
    end
    loss_curves{si} = loss_history;

    if isempty(best_U)
        [~, ~, fwd] = dlfeval(@(p) model_gradients(...
            p, X_dl, A_dl, labels_onehot_dl, labeled_mask, ...
            cfg, log_det_A, K, d, cfg.flow.numBlocks), params);
        best_U = double(gather(extractdata(fwd.U)));
    end

    if strcmp(cfg.k_selection, 'fixed')
        best_k = cfg.num_packages;
        labels_pred = kmeans(best_U, best_k, 'Replicates', cfg.kmeans_reps, ...
                             'MaxIter', cfg.kmeans_iter, 'Start', 'plus', 'Display', 'off');
    else
        Kmin = cfg.k_min;
        Kmax = min(cfg.k_max, N-1);
        best_score = -inf;
        best_k = Kmin;
        for ktry = Kmin:Kmax
            lbl_try = kmeans(best_U, ktry, 'Replicates', 5, 'MaxIter', 300, 'Display', 'off');
            sil_try = mean(silhouette(best_U, lbl_try));
            if sil_try > best_score
                best_score = sil_try;
                best_k = ktry;
            end
        end
        labels_pred = kmeans(best_U, best_k, 'Replicates', cfg.kmeans_reps, ...
                             'MaxIter', cfg.kmeans_iter, 'Start', 'plus', 'Display', 'off');
    end
    result_labels{si} = labels_pred;

    sil = mean(silhouette(best_U, labels_pred));
    try; ev = evalclusters(best_U, labels_pred, 'CalinskiHarabasz'); ch = ev.CriterionValues; catch; ch = NaN; end
    try; ev = evalclusters(best_U, labels_pred, 'DaviesBouldin'); db = ev.CriterionValues; catch; db = NaN; end

    mq = compute_MQ(A_norm, labels_pred);

    A_union_bin = A;
    if has_gt
        mojofm = compute_MoJoFM(gt_labels, labels_pred);
        turbomq = compute_TurboMQ(A_union_bin, labels_pred);
        [coh, coup] = compute_cohesion_coupling(A_union_bin, labels_pred);
        nmi = computeNMI(gt_labels, labels_pred);
        ari = computeARI(gt_labels, labels_pred);
    else
        mojofm = NaN; turbomq = NaN; coh = NaN; coup = NaN; nmi = NaN; ari = NaN;
    end

    results(si, :) = [seed, best_k, sil, ch, db, mq, mojofm, turbomq, coh, coup, nmi, ari];

    if cfg.verbose
        fprintf('  k=%d | Sil=%.4f | CH=%.2f | DB=%.4f | MQ=%.4f | MoJoFM=%.4f | NMI=%.4f\n', ...
                best_k, sil, ch, db, mq, mojofm, nmi);
    end

    out_fn = fullfile(cfg.outdir, sprintf('gcflow_seed%d.mat', seed));
    save(out_fn, 'best_U', 'labels_pred', 'sil', 'ch', 'db', 'mq', ...
         'mojofm', 'turbomq', 'coh', 'coup', 'nmi', 'ari', 'best_k', 'cfg', 'loss_history');
end

fprintf('\n========== SUMMARY over %d seeds ==========\n', nSeeds);
fprintf('k selection method: %s\n', cfg.k_selection);
if strcmp(cfg.k_selection, 'fixed')
    fprintf('Fixed k = %d (number of packages)\n', cfg.num_packages);
end
fprintf('----------------------------------------------------------------\n');
metric_names = {'Seed','k','Silhouette','CH','DB','MQ','MoJoFM','TurboMQ','Cohesion','Coupling','NMI','ARI'};
metrics_data = results(:, 3:end);
for m = 1:size(metrics_data,2)
    vals = metrics_data(:,m);
    name = metric_names{m+2};
    if any(strcmp(name, {'DB','Coupling'}))
        best = min(vals);
    else
        best = max(vals);
    end
    fprintf('%-12s  mean=%.4f  std=%.4f  best=%.4f\n', name, mean(vals), std(vals), best);
end

summary_file = fullfile(cfg.outdir, 'gcflow_summary.mat');
save(summary_file, 'results', 'metric_names', 'cfg', 'result_labels', 'loss_curves');
T = array2table(results, 'VariableNames', metric_names);
writetable(T, fullfile(cfg.outdir, 'gcflow_results.csv'));
fprintf('\nResults saved to %s\n', cfg.outdir);

end

function M = glorot_init(rows, cols)
lim = sqrt(6 / (rows + cols));
M = (rand(rows, cols) * 2 - 1) * lim;
end

function [Z, U, logdet] = gcflow_forward_dl(W1, b1, W2, b2, flow, B, A_dl, X_dl, d)
N_ = size(X_dl, 1);
H1 = max(X_dl * W1' + b1, 0);
H1 = A_dl * H1;
Z = H1 * W2' + b2;
Z = A_dl * Z;
xcur = Z;
logdet = dlarray(zeros(N_, 1, 'like', X_dl));
for bb = 1:B
    blk = flow{bb};
    mask = blk.mask;
    X_tilde = A_dl * xcur;
    idx_a = find(mask == 1);
    idx_b = find(mask == 0);
    x_a = X_tilde(:, idx_a);
    x_b = X_tilde(:, idx_b);
    S_h = max(x_a * blk.s.W1' + blk.s.b1, 0);
    S = tanh(S_h * blk.s.W2' + blk.s.b2);
    T_h = max(x_a * blk.t.W1' + blk.t.b1, 0);
    T = T_h * blk.t.W2' + blk.t.b2;
    y_b = x_b .* exp(S) + T;
    logdet = logdet + sum(S, 2);
    x_new = dlarray(zeros(N_, d, 'like', X_dl));
    x_new(:, idx_a) = x_a;
    x_new(:, idx_b) = y_b;
    xcur = x_new;
end
U = xcur;
end

function log_base = compute_log_base_mixture(U, gmm_mean_scale, gmm_cov_scale, gmm_log_weights, K, d)
N_ = size(U,1);
log_base = dlarray(zeros(N_,1,'like',U));
for k = 1:K
    mu_k = gmm_mean_scale(k) * ones(1,d) / sqrt(d);
    cov_diag = exp(gmm_cov_scale(k)) * ones(1,d);
    log_phi = gmm_log_weights(k);
    diff = U - mu_k;
    log_nk = -0.5 * sum(diff.^2 ./ cov_diag, 2) ...
             -0.5 * d * log(2*pi) - 0.5 * sum(log(cov_diag));
    log_base = log_base + exp(log_phi + log_nk - log_base);
end
log_base = log(log_base + 1e-12);
end

function [loss, grads, fwd] = model_gradients(params_, X_dl_, A_dl_, labels_onehot_, labeled_mask_, cfg_, log_det_A_, K_, d_, B_)
N_ = size(X_dl_, 1);
[Z_dl_, U_dl_, logdet_dl_] = gcflow_forward_dl(...
    params_.W1, params_.b1, params_.W2, params_.b2, ...
    params_.flow, B_, A_dl_, X_dl_, d_);
log_base = compute_log_base_mixture(U_dl_, params_.gmm_mean_scale, ...
              params_.gmm_cov_scale, params_.gmm_log_weights, K_, d_);
log_jac = logdet_dl_;
log_det_term = B_ * d_ * log_det_A_ / N_;
log_px = log_base + log_jac + log_det_term;

labeled_idx = find(labeled_mask_);
n_labeled = length(labeled_idx);
if n_labeled > 0
    labels_onehot_double = extractdata(labels_onehot_);
    log_pxy = dlarray(zeros(N_,1,'like',U_dl_));
    for i = 1:n_labeled
        idx = labeled_idx(i);
        true_k = find(labels_onehot_double(i,:), 1);
        if isempty(true_k), continue; end
        mu_k = params_.gmm_mean_scale(true_k) * ones(1,d_) / sqrt(d_);
        cov_diag = exp(params_.gmm_cov_scale(true_k)) * ones(1,d_);
        diff = U_dl_(idx,:) - mu_k;
        log_n = -0.5 * sum(diff.^2 ./ cov_diag, 2) ...
                -0.5 * d_ * log(2*pi) - 0.5 * sum(log(cov_diag));
        log_pxy(idx) = log_n + params_.gmm_log_weights(true_k) + logdet_dl_(idx) + log_det_term;
    end
    loss_labeled = -mean(log_pxy(labeled_idx));
else
    loss_labeled = 0;
end

unlabeled_idx = find(~labeled_mask_);
n_unlabeled = length(unlabeled_idx);
if n_unlabeled > 0
    loss_unlabeled = -mean(log_px(unlabeled_idx));
else
    loss_unlabeled = 0;
end

lambda = cfg_.lambda_unlabeled;
loss = (1 - lambda) * loss_labeled + lambda * loss_unlabeled;

plist = params_to_list(params_, B_);
grads = dlgradient(loss, plist, 'EnableHigherDerivatives', false);

fwd.Z = Z_dl_;
fwd.U = U_dl_;
end

function plist = params_to_list(p_, B_)
plist = {p_.W1, p_.b1, p_.W2, p_.b2};
for b = 1:B_
    blk = p_.flow{b};
    plist = [plist, {blk.s.W1, blk.s.b1, blk.s.W2, blk.s.b2, ...
                     blk.t.W1, blk.t.b1, blk.t.W2, blk.t.b2}];
end
plist = [plist, {p_.gmm_mean_scale, p_.gmm_cov_scale, p_.gmm_log_weights}];
end

function p_ = pack_params_dl(W1, b1, W2, b2, flow, B, gmm_mean_scale, gmm_cov_scale, gmm_log_weights)
p_.W1 = dlarray(W1);
p_.b1 = dlarray(b1);
p_.W2 = dlarray(W2);
p_.b2 = dlarray(b2);
p_.flow = cell(B,1);
for b = 1:B
    blk = flow{b};
    bdl.s.W1 = dlarray(blk.s.W1); bdl.s.b1 = dlarray(blk.s.b1);
    bdl.s.W2 = dlarray(blk.s.W2); bdl.s.b2 = dlarray(blk.s.b2);
    bdl.t.W1 = dlarray(blk.t.W1); bdl.t.b1 = dlarray(blk.t.b1);
    bdl.t.W2 = dlarray(blk.t.W2); bdl.t.b2 = dlarray(blk.t.b2);
    bdl.mask = blk.mask;
    p_.flow{b} = bdl;
end
p_.gmm_mean_scale = dlarray(gmm_mean_scale);
p_.gmm_cov_scale  = dlarray(gmm_cov_scale);
p_.gmm_log_weights = dlarray(gmm_log_weights);
end

function state = init_adam_state(p_)
plist = params_to_list(p_, numel(p_.flow));
state.m = cell(size(plist));
state.v = cell(size(plist));
state.t = 0;
for i = 1:numel(plist)
    state.m{i} = zeros(size(plist{i}), 'like', plist{i});
    state.v{i} = zeros(size(plist{i}), 'like', plist{i});
end
end

function [p_out, state_out] = adam_step(p_, grads, state, lr, wd)
beta1 = 0.9; beta2 = 0.999; eps_adam = 1e-8;
B_loc = numel(p_.flow);
plist = params_to_list(p_, B_loc);
state_out = state;
state_out.t = state.t + 1;
t_ = state_out.t;

new_plist = cell(size(plist));
for i = 1:numel(plist)
    g = grads{i} + wd * plist{i};
    m_new = beta1 * state.m{i} + (1 - beta1) * g;
    v_new = beta2 * state.v{i} + (1 - beta2) * (g .* g);
    state_out.m{i} = m_new;
    state_out.v{i} = v_new;
    m_hat = m_new / (1 - beta1^t_);
    v_hat = v_new / (1 - beta2^t_);
    new_plist{i} = plist{i} - lr * (m_hat ./ (sqrt(v_hat) + eps_adam));
end

p_out = p_;
p_out.W1 = new_plist{1};
p_out.b1 = new_plist{2};
p_out.W2 = new_plist{3};
p_out.b2 = new_plist{4};
idx = 5;
for b = 1:B_loc
    blk = p_out.flow{b};
    blk.s.W1 = new_plist{idx}; idx=idx+1;
    blk.s.b1 = new_plist{idx}; idx=idx+1;
    blk.s.W2 = new_plist{idx}; idx=idx+1;
    blk.s.b2 = new_plist{idx}; idx=idx+1;
    blk.t.W1 = new_plist{idx}; idx=idx+1;
    blk.t.b1 = new_plist{idx}; idx=idx+1;
    blk.t.W2 = new_plist{idx}; idx=idx+1;
    blk.t.b2 = new_plist{idx}; idx=idx+1;
    p_out.flow{b} = blk;
end
p_out.gmm_mean_scale = new_plist{idx}; idx=idx+1;
p_out.gmm_cov_scale  = new_plist{idx}; idx=idx+1;
p_out.gmm_log_weights = new_plist{idx};
end

function grads_out = clip_gradients(grads_, clip_val)
grads_out = cell(size(grads_));
for i = 1:numel(grads_)
    g = grads_{i};
    grads_out{i} = max(min(g, clip_val), -clip_val);
end
end

function ld = compute_log_abs_det(A_n)
ev = eig(full(A_n));
ev = real(ev);
ev = ev(abs(ev) > 1e-12);
if isempty(ev)
    ld = -inf;
else
    ld = sum(log(abs(ev)));
end
end

function mq = compute_MQ(A_bin, clusters)
A_bin = double(A_bin > 0);
uq = unique(clusters);
K = numel(uq);
mqk = zeros(K,1);
for k = 1:K
    idx = find(clusters == uq(k));
    other = find(clusters ~= uq(k));
    I = sum(A_bin(idx,idx), 'all');
    E = sum(A_bin(idx,other), 'all');
    if I+E > 0
        mqk(k) = I/(I+E);
    end
end
mq = mean(mqk);
end

function mfm = compute_MoJoFM(true_labels, pred_labels)
true_labels = true_labels(:);
pred_labels = pred_labels(:);
n = length(true_labels);
[~,~,true_idx] = unique(true_labels);
[~,~,pred_idx] = unique(pred_labels);
Kt = max(true_idx);
Kp = max(pred_idx);
C = accumarray([true_idx, pred_idx], 1, [Kt, Kp]);
[vals, idx] = sort(C(:), 'descend');
matched = zeros(Kt,1);
used = false(1,Kp);
for i = 1:numel(vals)
    if vals(i) == 0, break; end
    [r,c] = ind2sub([Kt, Kp], idx(i));
    if ~matched(r) && ~used(c)
        matched(r) = c;
        used(c) = true;
    end
end
moved = 0;
for t = 1:Kt
    sz = sum(true_idx == t);
    if matched(t) > 0
        moved = moved + (sz - C(t, matched(t)));
    else
        moved = moved + sz;
    end
end
mfm = 1 - moved/n;
end

function tmq = compute_TurboMQ(A_bin, clusters)
A_bin = double(A_bin > 0);
uq = unique(clusters);
K = numel(uq);
mqk = zeros(K,1);
w = zeros(K,1);
for k = 1:K
    idx = find(clusters == uq(k));
    other = find(clusters ~= uq(k));
    I = sum(A_bin(idx,idx), 'all');
    E = sum(A_bin(idx,other), 'all');
    if I+E > 0
        mqk(k) = I/(I+E);
    end
    w(k) = length(idx);
end
tmq = sum(w.*mqk) / sum(w);
end

function [coh, coup] = compute_cohesion_coupling(A_bin, clusters)
A_bin = double(A_bin > 0);
uq = unique(clusters);
K = numel(uq);
coh = zeros(K,1);
coup = zeros(K,1);
for k = 1:K
    idx = find(clusters == uq(k));
    other = find(clusters ~= uq(k));
    n = length(idx);
    if n <= 1
        coh(k)=0; coup(k)=0;
        continue;
    end
    I = sum(A_bin(idx,idx), 'all');
    E = sum(A_bin(idx,other), 'all');
    coh(k) = I / (n*(n-1));
    if I+E > 0
        coup(k) = E / (I+E);
    else
        coup(k) = 0;
    end
end
coh = mean(coh);
coup = mean(coup);
end

function nmi = computeNMI(true_labels, pred_labels)
true_labels = true_labels(:);
pred_labels = pred_labels(:);
n = length(true_labels);
[cl, ~, true_idx] = unique(true_labels);
[ck, ~, pred_idx] = unique(pred_labels);
nc = numel(cl);
nk = numel(ck);
C = zeros(nc, nk);
for i = 1:n
    C(true_idx(i), pred_idx(i)) = C(true_idx(i), pred_idx(i)) + 1;
end
Ni = sum(C,2);
Nj = sum(C,1);
MI = 0;
for i = 1:nc
    for j = 1:nk
        if C(i,j) > 0
            MI = MI + (C(i,j)/n) * log(C(i,j)*n/(Ni(i)*Nj(j)));
        end
    end
end
Hi = -sum((Ni/n).*log(Ni/n+eps));
Hj = -sum((Nj/n).*log(Nj/n+eps));
nmi = max(0, MI / (sqrt(Hi*Hj)+eps));
end

function ari = computeARI(true_labels, pred_labels)
true_labels = true_labels(:);
pred_labels = pred_labels(:);
n = length(true_labels);
[cl, ~, true_idx] = unique(true_labels);
[ck, ~, pred_idx] = unique(pred_labels);
nc = numel(cl);
nk = numel(ck);
C = zeros(nc, nk);
for i = 1:n
    C(true_idx(i), pred_idx(i)) = C(true_idx(i), pred_idx(i)) + 1;
end
sc = sum(C(:).*(C(:)-1)/2);
rs = sum(C,2);
cs = sum(C,1);
sca = sum(rs.*(rs-1)/2);
scb = sum(cs.*(cs-1)/2);
exp_ = (sca * scb) / (n*(n-1)/2);
mx = (sca + scb)/2;
if abs(mx - exp_) < eps
    ari = 1;
else
    ari = (sc - exp_) / (mx - exp_);
end
end