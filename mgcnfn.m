function mgcnfn(cfg_in)

if nargin < 1 || isempty(cfg_in)
    cfg = struct();
else
    cfg = cfg_in;
end

if ~isfield(cfg,'A_paths')
    cfg.A_paths = {
        './javacc/adjacency_matrix_aggregation.txt',  ...
        './javacc/adjacency_matrix_association.txt',  ...
        './javacc/adjacency_matrix_composition.txt',  ...
        './javacc/adjacency_matrix_dependency.txt'    
    };
end
if ~isfield(cfg,'rel_names')
    cfg.rel_names = {'Aggregation','Association','Composition','Dependency'};
end
if ~isfield(cfg,'features'), cfg.features = './javacc/node_features.txt'; end
if ~isfield(cfg,'labels'),   cfg.labels   = './javacc/label.txt';         end

if ~isfield(cfg,'encoder_dims'), cfg.encoder_dims = [512, 256]; end
if ~isfield(cfg,'latent_dim'),   cfg.latent_dim   = 64;         end
if ~isfield(cfg,'gcn_hidden'),   cfg.gcn_hidden   = cfg.latent_dim; end
if ~isfield(cfg,'gcn_layers'),   cfg.gcn_layers   = 2;          end
if ~isfield(cfg,'epsilon'),      cfg.epsilon       = 0.5;        end
if ~isfield(cfg,'att_h'),        cfg.att_h         = 64;         end
if ~isfield(cfg,'alpha'),        cfg.alpha         = 1.0;        end
if ~isfield(cfg,'beta'),         cfg.beta          = 1e-4;       end

assert(cfg.latent_dim == cfg.gcn_hidden, ...
    '[CFG] latent_dim (%d) must equal gcn_hidden (%d).', ...
    cfg.latent_dim, cfg.gcn_hidden);
d = cfg.gcn_hidden;

if ~isfield(cfg,'use_gated_fusion'), cfg.use_gated_fusion = false;  end
if ~isfield(cfg,'use_actnorm'),      cfg.use_actnorm      = false; end

if ~isfield(cfg,'maxEpochs'),    cfg.maxEpochs    = 300;  end
if ~isfield(cfg,'lr'),           cfg.lr           = 1e-3; end
if ~isfield(cfg,'grad_clip'),    cfg.grad_clip    = 1.0;  end
if ~isfield(cfg,'verbose'),      cfg.verbose      = true; end
if ~isfield(cfg,'log_interval'), cfg.log_interval = 10;   end
if ~isfield(cfg,'seed_start'),   cfg.seed_start   = 1;    end
if ~isfield(cfg,'seed_end'),     cfg.seed_end     = 1;    end

if ~isfield(cfg,'flow')
    cfg.flow.numBlocks     = 4;
    cfg.flow.hidden        = 64;
    cfg.flow.mp_r          = 32;
    cfg.flow.mp_h          = 64;
    cfg.flow.att_h         = 32;
    cfg.flow.lambda        = 0.01;
    cfg.flow.freeze_epochs = 20;
    cfg.flow.logdet_clip   = 5;
    cfg.flow.z_reg         = 0;
    cfg.flow.struct_lambda = 0.1;
    cfg.flow.gcl_lambda    = 1.0;
    cfg.flow.gcl_pos       = 5;
    cfg.flow.gcl_neg       = 10;
    cfg.flow.gcl_tau       = 0.1;
    cfg.flow.modbal_lambda = 1.0;
end

if ~isfield(cfg,'gmm')
    cfg.gmm.K         = 10;
    cfg.gmm.diagCov   = true;
    cfg.gmm.init_scale= 1.0;
end

if ~isfield(cfg,'k_selection_max'),    cfg.k_selection_max    = 20;    end
if ~isfield(cfg,'k_selection_method'), cfg.k_selection_method = 'both'; end

if ~isfield(cfg,'outdir')
    cfg.outdir = fullfile(pwd,'mgccn_gnf_results');
end
if ~exist(cfg.outdir,'dir'), mkdir(cfg.outdir); end

ts      = datestr(now,'yyyymmdd_HHMMSS');
logfile = fullfile(cfg.outdir, sprintf('log_%s.txt', ts));
flog    = fopen(logfile, 'w');
write_log_header(flog, cfg);

V      = numel(cfg.A_paths);
graphs = cell(1, V);
for v = 1:V
    Av = double(load(cfg.A_paths{v}));
    if size(Av,1) ~= size(Av,2)
        error('Adjacency matrix %d is not square.', v);
    end
    graphs{v} = Av;
end

X         = double(load(cfg.features));
[N, D_raw]= size(X);
labels    = [];
if isfile(cfg.labels)
    labels = double(load(cfg.labels));
    labels = labels(:);
end

fprintf('Data loaded: V=%d, N=%d, D_raw=%d\n', V, N, D_raw);
fprintf(flog, 'Data loaded: V=%d, N=%d, D_raw=%d\n', V, N, D_raw);

mu_X    = mean(X, 1);
sigma_X = std(X, 0, 1) + 1e-8;
X       = (X - mu_X) ./ sigma_X;

freq     = sum(X > 0, 1);
n_remove = sum(freq < 5 | freq > 0.9*N);
fprintf('Preprocessing: %d rare/frequent features detected (not removed)\n', n_remove);
fprintf(flog, 'Preprocessing: %d rare/frequent features detected (not removed)\n', n_remove);

mu_pca   = mean(X, 1);
X_cen    = X - mu_pca;
[eigvec, eigval_mat] = eig(cov(X_cen));
eigvals  = diag(eigval_mat);

perm_ev = zeros(100, 1);
for i = 1:100
    perm_ev(i) = max(eig(cov(X_cen(randperm(N),:))));
end
mc_thr  = prctile(perm_ev, 75);
keep_idx= eigvals > mc_thr;
W_pca   = eigvec(:, keep_idx);
X       = X_cen * W_pca;
D       = size(X, 2);

if D < 20
    [~, si] = sort(eigvals, 'descend');
    W_pca   = eigvec(:, si(1:min(20, end)));
    X       = X_cen * W_pca;
    D       = size(X, 2);
end

fprintf('PCA: D_raw=%d -> D=%d (threshold=%.4f)\n', D_raw, D, mc_thr);
fprintf(flog, 'PCA: D_raw=%d -> D=%d (threshold=%.4f)\n\n', D_raw, D, mc_thr);

K_knn      = 4;
A_knn      = build_knn(X, K_knn);
graphs_all = [{A_knn}, graphs];
V_all      = numel(graphs_all);
cfg.num_rel_total = V_all;

A_union = zeros(N, N);
for v = 1:V_all
    Av_raw  = (graphs_all{v} + graphs_all{v}') / 2;
    A_union = A_union + Av_raw;
end
A_union = double(A_union > 0);

A_per_rel = cell(1, V);
for v = 1:V
    Av_raw      = (graphs{v} + graphs{v}') / 2;
    A_per_rel{v}= double(Av_raw > 0);
end

Atilde = cell(1, V_all);
for v = 1:V_all
    Av = graphs_all{v};
    Av = (Av + Av') / 2;
    Av = Av - diag(diag(Av));
    Av = Av + eye(N);
    deg = sum(Av, 2);
    Dm  = diag(1 ./ sqrt(max(deg, eps)));
    Atilde{v} = Dm * Av * Dm;
end

fprintf('KNN K=%d, V_all=%d\n\n', K_knn, V_all);
fprintf(flog, 'KNN K=%d, V_all=%d\n\n', K_knn, V_all);

fprintf('=== GRAPH STRUCTURE DIAGNOSIS ===\n');
fprintf(flog, '=== GRAPH STRUCTURE DIAGNOSIS ===\n');
fprintf('  %-14s  %6s  %9s  %9s  %6s\n', 'Relation','Edges','Density','Components','Isolated');
fprintf(flog,'  %-14s  %6s  %9s  %9s  %6s\n', 'Relation','Edges','Density','Components','Isolated');
for v = 1:V
    Av_d    = A_per_rel{v};
    n_edge  = sum(Av_d(:)) / 2;
    density = sum(Av_d(:)) / max(N*(N-1), 1);
    try
        G_v    = graph(Av_d);
        bins_v = conncomp(G_v);
        n_comp = max(bins_v);
        csz    = histcounts(bins_v, 1:n_comp+1);
        n_isol = sum(csz == 1);
    catch
        n_comp = NaN; n_isol = NaN;
    end
    fprintf('  %-14s  %6d  %9.5f  %9d  %6d\n', ...
        cfg.rel_names{v}, n_edge, density, n_comp, n_isol);
    fprintf(flog,'  %-14s  %6d  %9.5f  %9d  %6d\n', ...
        cfg.rel_names{v}, n_edge, density, n_comp, n_isol);
end
fprintf('Note: MQ=1.0 is usually caused by a disconnected graph\n');
fprintf('      (no cross-cluster edges). See Components column.\n');
fprintf('================================\n\n');
fprintf(flog,'Note: MQ=1.0 is usually caused by a disconnected graph.\n');
fprintf(flog,'================================\n\n');

enc_sizes = [D, cfg.encoder_dims, d];
Lenc      = numel(enc_sizes) - 1;
dec_sizes = [d, fliplr(cfg.encoder_dims), D];
Ldec      = numel(dec_sizes) - 1;
gcn_L     = cfg.gcn_layers;

n_fixed_metrics = 9;
n_rel_metrics   = V;
n_total_metrics = n_fixed_metrics + n_rel_metrics + 1;

nama_metrik = [{'Sil','CH','DB','MQ_std','MQ_hyb','k_sel','NMI','ARI','ExecTime_s'}, ...
               arrayfun(@(v)sprintf('MQ_%s',cfg.rel_names{v}),1:V,'UniformOutput',false), ...
               {'MQ_rel_dominan'}];

nSeeds     = cfg.seed_end - cfg.seed_start + 1;
blok_hasil = zeros(nSeeds, n_total_metrics);
seed_idx   = 0;

for seed = cfg.seed_start : cfg.seed_end
    seed_idx   = seed_idx + 1;
    t_seed_start = tic;

    fprintf('\n========== SEED %d (%d/%d) ==========\n', seed, seed_idx, nSeeds);
    fprintf(flog, '\n========== SEED %d (%d/%d) ==========\n', seed, seed_idx, nSeeds);
    rng(seed, 'twister');

    params = struct();

    params.enc = struct();
    for l = 1:Lenc
        params.enc.(['W' num2str(l)]) = dlarray(init_glorot(enc_sizes(l+1), enc_sizes(l)));
        params.enc.(['b' num2str(l)]) = dlarray(zeros(enc_sizes(l+1), 1));
    end

    params.dec = struct();
    for l = 1:Ldec
        params.dec.(['W' num2str(l)]) = dlarray(init_glorot(dec_sizes(l+1), dec_sizes(l)));
        params.dec.(['b' num2str(l)]) = dlarray(zeros(dec_sizes(l+1), 1));
    end

    params.gcn_knn = cell(1, gcn_L);
    for l = 1:gcn_L
        params.gcn_knn{l}.W = dlarray(init_glorot(d, d));
        params.gcn_knn{l}.b = dlarray(zeros(d, 1));
    end

    params.gcn = cell(1, V);
    for v = 1:V
        params.gcn{v} = cell(1, gcn_L);
        for l = 1:gcn_L
            inS = D*(l==1) + d*(l>1);
            params.gcn{v}{l}.W = dlarray(init_glorot(d, inS));
            params.gcn{v}{l}.b = dlarray(zeros(d, 1));
        end
    end

    params.sg_att = cell(1, V_all);
    for v = 1:V_all
        params.sg_att{v}.w = dlarray(init_glorot(d, 1)');
        params.sg_att{v}.b = dlarray(0);
    end

    params.mg_att.W = dlarray(init_glorot(cfg.att_h, d));
    params.mg_att.b = dlarray(zeros(cfg.att_h, 1));
    params.mg_att.q = dlarray(init_glorot(cfg.att_h, 1));

    if cfg.use_gated_fusion
        params.gf.Wg = dlarray(init_glorot(d, d * V_all));
        params.gf.bg = dlarray(zeros(d, 1));
    end

    params.flow = init_flow_params(cfg.flow, d, V_all, cfg.use_actnorm);
    params.gmm  = init_gmm_params(cfg.gmm, d);

    [paramList, paramNames] = flatten_params(params, V, gcn_L, V_all, cfg);
    numP   = numel(paramList);
    beta1  = 0.9;  beta2 = 0.999;  eps_adam = 1e-8;
    m_ad   = cell(1, numP);
    v_ad   = cell(1, numP);
    for i = 1:numP
        sz      = size(extractdata(paramList{i}));
        m_ad{i} = zeros(sz);
        v_ad{i} = zeros(sz);
    end
    Xdl = dlarray(X);

    fprintf(flog, '--- Training ---\n');
    lossHistory = zeros(cfg.maxEpochs, 1);
    lossComp    = zeros(cfg.maxEpochs, 6);
    iter        = 0;
    cur_lr      = cfg.lr;

    for epoch = 1:cfg.maxEpochs
        t_ep = tic;
        cfg.currentEpoch = epoch;

        [lossVal, grads, outs] = dlfeval( ...
            @(p) model_gradients(p, Atilde, Xdl, cfg, N, D, V, V_all, gcn_L, d, Lenc), ...
            params);

        lossNum            = double(gather(extractdata(lossVal)));
        lossHistory(epoch) = lossNum;
        if isfield(outs, 'loss_components')
            lossComp(epoch, :) = outs.loss_components;
        end

        for gi = 1:numP
            if ~isempty(grads{gi})
                grads{gi} = max(min(grads{gi}, cfg.grad_clip), -cfg.grad_clip);
            end
        end

        iter = iter + 1;
        for p = 1:numP
            g = grads{p};
            if isempty(g), continue; end
            if ~isa(g,'dlarray'), g = dlarray(double(g)); end
            gd       = double(extractdata(g));
            m_ad{p}  = beta1 * m_ad{p}  + (1 - beta1) * gd;
            v_ad{p}  = beta2 * v_ad{p}  + (1 - beta2) * gd.^2;
            mhat     = m_ad{p}  / (1 - beta1^iter);
            vhat     = v_ad{p}  / (1 - beta2^iter);
            step     = cur_lr * (mhat ./ (sqrt(vhat) + eps_adam));
            paramList{p} = dlarray(double(extractdata(paramList{p})) - step);
        end
        params = unflatten_params(paramList, paramNames, cfg, V, gcn_L, V_all);

        if cfg.verbose && (mod(epoch, cfg.log_interval)==0 || epoch==1)
            fprintf('  Ep %3d/%d  Loss=%.5e  t=%.2fs\n', ...
                epoch, cfg.maxEpochs, lossNum, toc(t_ep));
            fprintf(flog, '  Ep %3d  Loss=%.5e\n', epoch, lossNum);
        end

        if mod(epoch, 50) == 0
            cur_lr = cur_lr * 0.5;
        end
    end

    [~, ~, outFinal] = dlfeval( ...
        @(p) model_gradients(p, Atilde, Xdl, cfg, N, D, V, V_all, gcn_L, d, Lenc), ...
        params);
    Z_fused = double(gather(extractdata(outFinal.Z_fused)));
    U       = double(gather(extractdata(outFinal.U)));
    Xrec    = double(gather(extractdata(outFinal.Xrec)));
    exec_time = toc(t_seed_start);

    Emb  = U;
    Kmin = 3;
    Kmax = cfg.k_selection_max;
    if ~isempty(labels)
        Kmax = max(Kmax, numel(unique(labels)) * 2);
    end
    Ktry = Kmin : min(Kmax, N-1);

    best_k    = Kmin;
    best_score= -inf;
    for ktest = Ktry
        try
            lbl = kmeans(Emb, ktest, 'Replicates', 5, 'MaxIter', 300, 'Start', 'plus');
        catch
            lbl = kmeans(Emb, ktest, 'Replicates', 3, 'MaxIter', 300);
        end
        sil_k = safe_silhouette(Emb, lbl);
        mq_k  = compute_MQ(A_union, lbl);
        switch lower(cfg.k_selection_method)
            case 'silhouette', score = sil_k;
            case 'mq',         score = mq_k;
            otherwise,         score = sil_k + 1e-4 * mq_k;
        end
        if score > best_score
            best_score = score;
            best_k     = ktest;
        end
    end

    try
        labels_pred = kmeans(Emb, best_k, 'Replicates', 20, 'MaxIter', 500, 'Start', 'plus');
    catch
        labels_pred = kmeans(Emb, best_k, 'Replicates', 10, 'MaxIter', 500);
    end

    silVal  = safe_silhouette(Emb, labels_pred);
    try,  chVal = evalclusters(Emb, labels_pred, 'CalinskiHarabasz').CriterionValues;
    catch, chVal = NaN; end
    try,  dbVal = evalclusters(Emb, labels_pred, 'DaviesBouldin').CriterionValues;
    catch, dbVal = NaN; end

    [MQ_std, mq_nact_u, mq_nzero_u] = compute_MQ(A_union, labels_pred);
    [MQ_hyb, ~, ~]                  = compute_MQ_hybrid(A_union, Z_fused, labels_pred);

    nmiVal = 0; ariVal = 0;
    if ~isempty(labels)
        nmiVal = computeNMI(labels, labels_pred);
        ariVal = computeARI(labels, labels_pred);
    end

    MQ_per_rel   = zeros(1, V);
    mq_nact_rel  = zeros(1, V);
    mq_nzero_rel = zeros(1, V);
    for v = 1:V
        [MQ_per_rel(v), mq_nact_rel(v), mq_nzero_rel(v)] = ...
            compute_MQ(A_per_rel{v}, labels_pred);
    end
    [~, idx_dom]   = max(MQ_per_rel);
    rel_dominan_nm = cfg.rel_names{idx_dom};

    fprintf('\n  Seed %d results:\n', seed);
    fprintf('    k=%d  Sil=%.4f  CH=%.2f  DB=%.4f\n', best_k, silVal, chVal, dbVal);
    fprintf('    MQ_std=%.4f  MQ_hyb=%.4f  NMI=%.4f  ARI=%.4f\n', MQ_std, MQ_hyb, nmiVal, ariVal);
    fprintf('    MQ_std=%.4f (active=%d/no-edge=%d)  MQ_hyb=%.4f\n', ...
        MQ_std, mq_nact_u, mq_nzero_u, MQ_hyb);
    fprintf('    NMI=%.4f  ARI=%.4f\n', nmiVal, ariVal);
    fprintf('    MQ per relation (active clusters / total):\n');
    for v = 1:V
        flag = '';
        if MQ_per_rel(v) >= 0.9999
            flag = ' [WARNING: MQ=1.0, possible disconnected graph]';
        end
        fprintf('      %-14s : %.4f  (active=%d, no-edge=%d)%s\n', ...
            cfg.rel_names{v}, MQ_per_rel(v), mq_nact_rel(v), mq_nzero_rel(v), flag);
    end
    fprintf('    Dominant relation: %s (MQ=%.4f)\n', rel_dominan_nm, MQ_per_rel(idx_dom));
    fprintf('    Execution time: %.2f s\n', exec_time);

    fprintf(flog, '\n  Seed %d results:\n', seed);
    fprintf(flog, '    k=%d  Sil=%.4f  CH=%.4f  DB=%.4f\n', best_k, silVal, chVal, dbVal);
    fprintf(flog, '    MQ_std=%.4f (active=%d, no-edge=%d)\n', MQ_std, mq_nact_u, mq_nzero_u);
    fprintf(flog, '    MQ_hyb=%.4f  NMI=%.4f  ARI=%.4f\n', MQ_hyb, nmiVal, ariVal);
    fprintf(flog, '    MQ per relation:\n');
    for v = 1:V
        flag = '';
        if MQ_per_rel(v) >= 0.9999
            flag = ' [disconnected or fully intra-cluster graph]';
        end
        fprintf(flog, '      %-14s : %.4f  (active=%d, no-edge=%d)%s\n', ...
            cfg.rel_names{v}, MQ_per_rel(v), mq_nact_rel(v), mq_nzero_rel(v), flag);
    end
    fprintf(flog, '    Dominant relation: %s (MQ=%.4f)\n', rel_dominan_nm, MQ_per_rel(idx_dom));
    fprintf(flog, '    Time     : %.2f s\n', exec_time);
    fprintf(flog, '    Final loss: %.6e\n', lossHistory(end));

    blok_hasil(seed_idx, :) = [silVal, chVal, dbVal, MQ_std, MQ_hyb, best_k, ...
                                nmiVal, ariVal, exec_time, MQ_per_rel, idx_dom];

    save(fullfile(cfg.outdir, sprintf('run_seed%d.mat', seed)), ...
        'params', 'cfg', 'lossHistory', 'lossComp', ...
        'Z_fused', 'U', 'Xrec', 'labels_pred', ...
        'silVal', 'chVal', 'dbVal', 'MQ_std', 'MQ_hyb', ...
        'nmiVal', 'ariVal', 'best_k', 'exec_time', ...
        'MQ_per_rel', 'idx_dom', 'rel_dominan_nm');

    make_publication_figures(Emb, Z_fused, labels_pred, labels, ...
        lossHistory, lossComp, A_union, A_per_rel, Atilde, cfg, seed, ...
        silVal, MQ_std, MQ_hyb, nmiVal, ariVal, best_k, MQ_per_rel);

end

fprintf('\n\n===== SUMMARY OF %d SEEDS =====\n', nSeeds);
fprintf(flog, '\n\n===== SUMMARY OF %d SEEDS =====\n', nSeeds);
hdr = '%-18s  %8s  %8s  %8s\n';
row = '%-18s  %8.4f  %8.4f  %8.4f\n';
fprintf(hdr,     'Metric', 'Mean', 'Std', 'Max');
fprintf(flog, hdr,'Metric', 'Mean', 'Std', 'Max');
for m = 1:n_total_metrics
    col = blok_hasil(:, m);
    fprintf(row,     nama_metrik{m}, mean(col), std(col), max(col));
    fprintf(flog, row, nama_metrik{m}, mean(col), std(col), max(col));
end

fprintf(flog, '\n--- Raw Data ---\n%-5s', 'Seed');
for m = 1:n_total_metrics
    fprintf(flog, '  %-14s', nama_metrik{m});
end
fprintf(flog, '\n');
for s = 1:nSeeds
    fprintf(flog, '%-5d', cfg.seed_start + s - 1);
    for m = 1:n_total_metrics
        fprintf(flog, '  %-14.6f', blok_hasil(s,m));
    end
    fprintf(flog, '\n');
end

save(fullfile(cfg.outdir,'summary.mat'), 'blok_hasil', 'nama_metrik', 'cfg');
fclose(flog);
fprintf('\nDone. Log saved to: %s\n', logfile);

    function [loss, gradsOut, outputs] = model_gradients( ...
            paramsL, AtildeL, XdlL, cfgL, Nloc, ~, Vloc, V_all_loc, gcn_L_loc, d_loc, Lenc_loc)

        H = XdlL;
        He_layers = cell(1, Lenc_loc);
        for ll = 1:Lenc_loc
            W = paramsL.enc.(['W' num2str(ll)]);
            b = paramsL.enc.(['b' num2str(ll)]);
            H = H * W' + b';
            if ll < Lenc_loc, H = relu(H); end
            He_layers{ll} = H;
        end
        He = H;

        Xhat = He;
        Ldec_loc = numel(fieldnames(paramsL.dec)) / 2;
        for ll = 1:Ldec_loc
            Wd   = paramsL.dec.(['W' num2str(ll)]);
            bd   = paramsL.dec.(['b' num2str(ll)]);
            Xhat = Xhat * Wd' + bd';
            if ll < Ldec_loc, Xhat = relu(Xhat); end
        end
        Xrec = Xhat;

        Z_knn = He;
        Aknn_t = AtildeL{1};
        for ll = 1:gcn_L_loc
            Z_tilde = (1 - cfgL.epsilon) * Z_knn + cfgL.epsilon * He;
            Wk      = paramsL.gcn_knn{ll}.W;
            bk      = paramsL.gcn_knn{ll}.b;
            Z_knn   = relu((Aknn_t * Z_tilde) * Wk' + bk');
        end

        Zv = cell(1, Vloc);
        for vv = 1:Vloc
            Zcur = XdlL;
            At_v = AtildeL{vv+1};
            for ll = 1:gcn_L_loc
                Wg   = paramsL.gcn{vv}{ll}.W;
                bg   = paramsL.gcn{vv}{ll}.b;
                Zcur = relu((At_v * Zcur) * Wg' + bg');
            end
            Zv{vv} = Zcur;
        end

        Zhat = cell(1, V_all_loc);
        for vv = 1:V_all_loc
            if vv == 1, Z_in = Z_knn; else, Z_in = Zv{vv-1}; end
            wv    = paramsL.sg_att{vv}.w;
            bv    = paramsL.sg_att{vv}.b;
            s     = tanh(Z_in * wv' + bv);
            s_exp = exp(s - max(s, [], 1));
            alpha = s_exp ./ (sum(s_exp, 1) + eps);
            Zhat{vv} = sigmoid(alpha .* Z_in);
        end

        Wm    = paramsL.mg_att.W;
        bm    = paramsL.mg_att.b;
        qm    = paramsL.mg_att.q;
        Omega = dlarray(zeros(Nloc, V_all_loc, 'like', XdlL));
        for vv = 1:V_all_loc
            H_v          = tanh(Zhat{vv} * Wm' + bm');
            Omega(:, vv) = H_v * qm;
        end
        Ome_exp     = exp(Omega - max(Omega, [], 2));
        Alpha_graph = Ome_exp ./ (sum(Ome_exp, 2) + eps);
        Z_mgatt     = dlarray(zeros(Nloc, d_loc, 'like', XdlL));
        for vv = 1:V_all_loc
            Z_mgatt = Z_mgatt + Alpha_graph(:,vv) .* Zhat{vv};
        end

        if cfgL.use_gated_fusion && isfield(paramsL,'gf')
            Zcat   = Zhat{1};
            Z_mean = Zhat{1};
            for vv = 2:V_all_loc
                Zcat   = [Zcat,   Zhat{vv}];
                Z_mean = Z_mean + Zhat{vv};
            end
            Z_mean  = Z_mean / V_all_loc;
            g       = sigmoid(Zcat * paramsL.gf.Wg' + paramsL.gf.bg');
            Z_fused = g .* Z_mgatt + (1 - g) .* Z_mean;
        else
            Z_fused = Z_mgatt;
        end

        L_res = (1 / (2*Nloc)) * sum((XdlL - Xrec).^2, 'all');

        Sv = cell(1, V_all_loc);
        for vv = 1:V_all_loc
            Zt     = Zhat{vv};
            rn     = sqrt(sum(Zt.^2, 2) + eps);
            Sv{vv} = (Zt ./ rn) * (Zt ./ rn)';
        end
        Sknn = Sv{1};
        Lm   = dlarray(0);
        for vv = 2:V_all_loc
            dS = Sknn - Sv{vv};
            Lm = Lm + sum(dS.^2, 'all');
        end
        for i1 = 1:V_all_loc
            for j1 = (i1+1):V_all_loc
                dS = Sv{i1} - Sv{j1};
                Lm = Lm + 0.5 * sum(dS.^2, 'all');
            end
        end

        cur_ep    = 0;
        if isfield(cfgL,'currentEpoch'), cur_ep = cfgL.currentEpoch; end
        freeze_ep = cfgL.flow.freeze_epochs;

        if cur_ep <= freeze_ep
            U          = Z_fused;
            logdet_vec = dlarray(zeros(Nloc, 1, 'like', XdlL));
            L_flow     = dlarray(0);
            logp       = dlarray(zeros(Nloc, 1, 'like', XdlL));
        else
            [U, logdet_vec] = flow_forward(paramsL.flow, Z_fused, AtildeL, ...
                                            cfgL.flow, cfgL.use_actnorm);
            ld_clip    = cfgL.flow.logdet_clip;
            logdet_vec = min(max(logdet_vec, -ld_clip), ld_clip);
            logp       = gmm_logprob(paramsL.gmm, U);
            L_flow     = -mean(logp + logdet_vec);
        end

        z_reg   = cfgL.flow.z_reg;
        L_z_reg = z_reg * mean(sum(Z_fused.^2, 2));

        A_dl    = dlarray(AtildeL{1});
        dv      = sum(A_dl, 2);
        Lz      = (dv .* Z_fused) - (A_dl * Z_fused);
        L_struct= cfgL.flow.struct_lambda * 0.5 * ...
                  (sum(Z_fused .* Lz, 'all') / (sum(A_dl(:)) + eps));

        gcl_lam = cfgL.flow.gcl_lambda;
        L_gcl   = dlarray(0);
        if gcl_lam > 0
            Ppos  = cfgL.flow.gcl_pos;
            Pneg  = cfgL.flow.gcl_neg;
            tau   = cfgL.flow.gcl_tau;
            A_bin = double(AtildeL{1} > 0);
            Znorm = Z_fused ./ (sqrt(sum(Z_fused.^2, 2)) + eps);
            pos_sc= dlarray(zeros(0, 1, 'like', XdlL));
            neg_sc= dlarray(zeros(0, 1, 'like', XdlL));
            for i = 1:Nloc
                nbrs = find(A_bin(i,:) > 0);
                if isempty(nbrs), continue; end
                if numel(nbrs) > Ppos
                    nbrs = nbrs(randperm(numel(nbrs), Ppos));
                end
                zi = Znorm(i,:);
                for jj = 1:numel(nbrs)
                    j        = nbrs(jj);
                    pos_sc(end+1, 1) = sum(zi .* Znorm(j,:)) / tau;
                    neg_pool = setdiff(1:Nloc, [i, find(A_bin(i,:)>0)]);
                    if isempty(neg_pool), continue; end
                    if numel(neg_pool) > Pneg
                        neg_pool = neg_pool(randperm(numel(neg_pool), Pneg));
                    end
                    for kk = 1:numel(neg_pool)
                        neg_sc(end+1, 1) = sum(zi .* Znorm(neg_pool(kk),:)) / tau;
                    end
                end
            end
            lsig = @(x)(-log(1 + exp(-x) + eps));
            if numel(pos_sc) > 0 && numel(neg_sc) > 0
                L_gcl = gcl_lam * (-mean(lsig(pos_sc)) - mean(lsig(-neg_sc)));
            end
        end

        mb_lam   = cfgL.flow.modbal_lambda;
        L_modbal = dlarray(0);
        if mb_lam > 0
            R = V_all_loc - 1;
            if R == 4, w_rel = [2.0, 1.5, 3.0, 1.0]; else, w_rel = ones(1,R); end
            if isfield(cfgL.flow,'modbal_rel_weights') && numel(cfgL.flow.modbal_rel_weights)==R
                w_rel = cfgL.flow.modbal_rel_weights;
            end
            d_w = dlarray(zeros(Nloc, 1, 'like', XdlL));
            sw  = sum(w_rel);
            for rr = 1:R
                Ar          = double(AtildeL{rr+1});
                Ar(1:Nloc+1:end) = 0;
                deg_r       = sum(Ar, 2);
                m_r         = (Ar * Z_fused) ./ (deg_r + eps);
                d_r         = sqrt(sum((Z_fused - m_r).^2, 2));
                d_w         = d_w + w_rel(rr) * d_r;
            end
            d_w    = d_w / (sw + eps);
            d_mean = mean(d_w);
            L_cg   = mean(max(0, 1.0 * d_mean - d_w));
            mu_z   = mean(Z_fused, 1);
            sig2   = mean(sum((Z_fused - mu_z).^2, 2));
            L_var  = max(0, 0.5 - sig2);
            L_modbal = mb_lam * (0.01*L_cg + 0.005*L_var);
        end

        fl_lam = cfgL.flow.lambda;
        loss   = cfgL.alpha * L_res + cfgL.beta * Lm + fl_lam * L_flow + L_z_reg ...
               + L_struct + L_gcl + L_modbal;

        plist    = params_to_list(paramsL, Vloc, gcn_L_loc, V_all_loc, cfgL);
        gradsOut = dlgradient(loss, plist, 'EnableHigherDerivatives', false);

        outputs.Z_fused = Z_fused;
        outputs.U       = U;
        outputs.Xrec    = Xrec;
        outputs.logp    = logp;
        outputs.logdet  = logdet_vec;
        outputs.loss_components = [ ...
            double(gather(extractdata(cfgL.alpha * L_res))), ...
            double(gather(extractdata(cfgL.beta  * Lm))), ...
            double(gather(extractdata(fl_lam * L_flow))), ...
            double(gather(extractdata(L_struct))), ...
            double(gather(extractdata(L_gcl))), ...
            double(gather(extractdata(L_modbal)))];
    end

    function [Uout, logdet_acc] = flow_forward(flowS, Xin, AtildeL, flow_cfg, use_actnorm)
        Nloc       = size(Xin, 1);
        Dloc       = size(Xin, 2);
        xcur       = Xin;
        logdet_acc = dlarray(zeros(Nloc, 1, 'like', Xin));

        for bb = 1:numel(flowS.blocks)
            blk   = flowS.blocks{bb};
            mask  = double(blk.mask);
            idx_a = find(mask == 1);
            idx_b = find(mask == 0);

            if use_actnorm && isfield(blk,'an')
                an_s       = exp(blk.an.log_s);
                an_b       = blk.an.bias;
                xcur       = (xcur - an_b) .* an_s;
                logdet_acc = logdet_acc + sum(blk.an.log_s);
            end

            x_a = xcur(:, idx_a);
            x_b = xcur(:, idx_b);

            C = multi_graph_mp(x_a, AtildeL, blk.mp);

            S = relu(C * blk.snet.W1' + blk.snet.b1');
            S = S * blk.snet.W2' + blk.snet.b2';
            S = 1.5 * tanh(S);

            T = relu(C * blk.tnet.W1' + blk.tnet.b1');
            T = T * blk.tnet.W2' + blk.tnet.b2';

            y_b        = x_b .* exp(S) + T;
            logdet_acc = logdet_acc + sum(S, 2);

            x_new          = dlarray(zeros(Nloc, Dloc, 'like', Xin));
            x_new(:, idx_a)= x_a;
            x_new(:, idx_b)= y_b;
            xcur           = x_new;
        end
        Uout = xcur;
    end

    function C = multi_graph_mp(Xa, AtildeL, mp_params)
        Nloc = size(Xa, 1);
        Vloc = mp_params.num_rel;
        hmp  = size(mp_params.Wmp, 1);

        phiX   = relu(Xa * mp_params.Wphi' + mp_params.bphi');
        Hv_all = zeros(Nloc, hmp, Vloc, 'like', Xa);
        for vv = 1:Vloc
            m_v           = AtildeL{vv} * phiX;
            inp           = [Xa, m_v];
            Hv_all(:,:,vv)= relu(inp * mp_params.Wmp' + mp_params.bmp');
        end

        Ome = zeros(Nloc, Vloc, 'like', Xa);
        for vv = 1:Vloc
            att_in    = tanh(Hv_all(:,:,vv) * mp_params.Watt' + mp_params.batt');
            Ome(:,vv) = att_in * mp_params.u';
        end
        Ome_exp = exp(Ome - max(Ome, [], 2));
        Alpha   = Ome_exp ./ (sum(Ome_exp, 2) + eps);

        C = zeros(Nloc, hmp, 'like', Xa);
        for vv = 1:Vloc
            C = C + Alpha(:,vv) .* Hv_all(:,:,vv);
        end
        C = dlarray(C);
    end

    function flow = init_flow_params(flow_cfg, Dloc, Vall, use_actnorm)
        B    = flow_cfg.numBlocks;
        hid  = flow_cfg.hidden;
        r    = flow_cfg.mp_r;
        hmp  = flow_cfg.mp_h;
        att_h= flow_cfg.att_h;
        flow.blocks = cell(1, B);

        for b = 1:B
            blk = struct();
            if mod(b,2) == 1
                blk.mask = [ones(1,ceil(Dloc/2)),  zeros(1,floor(Dloc/2))];
            else
                blk.mask = [zeros(1,ceil(Dloc/2)), ones(1,floor(Dloc/2))];
            end
            dA             = ceil(Dloc/2);
            blk.mp.Wphi    = dlarray(init_glorot(r, dA));
            blk.mp.bphi    = dlarray(zeros(r, 1));
            blk.mp.Wmp     = dlarray(init_glorot(hmp, dA+r));
            blk.mp.bmp     = dlarray(zeros(hmp, 1));
            blk.mp.Watt    = dlarray(init_glorot(att_h, hmp));
            blk.mp.batt    = dlarray(zeros(att_h, 1));
            blk.mp.u       = dlarray(init_glorot(1, att_h));
            blk.mp.num_rel = Vall;
            dB             = floor(Dloc/2);
            blk.snet.W1    = dlarray(init_glorot(hid, hmp));
            blk.snet.b1    = dlarray(zeros(hid, 1));
            blk.snet.W2    = dlarray(init_glorot(dB, hid));
            blk.snet.b2    = dlarray(zeros(dB, 1));
            blk.tnet.W1    = dlarray(init_glorot(hid, hmp));
            blk.tnet.b1    = dlarray(zeros(hid, 1));
            blk.tnet.W2    = dlarray(init_glorot(dB, hid));
            blk.tnet.b2    = dlarray(zeros(dB, 1));
            if use_actnorm
                blk.an.log_s = dlarray(zeros(1, Dloc));
                blk.an.bias  = dlarray(zeros(1, Dloc));
            end
            flow.blocks{b} = blk;
        end
        flow.cfg = flow_cfg;
    end

    function gmm = init_gmm_params(gmm_cfg, Dloc)
        K          = gmm_cfg.K;
        gmm.K      = K;
        gmm.logits = dlarray(zeros(K, 1));
        gmm.mu     = dlarray(randn(Dloc, K) * 0.01);
        gmm.logvar = dlarray(zeros(Dloc, K));
        gmm.diagCov= gmm_cfg.diagCov;
    end

    function logp = gmm_logprob(gmmS, Uin)
        [Nloc, Dloc] = size(Uin);
        K    = gmmS.K;
        lse  = gmmS.logits(1) + log(sum(exp(gmmS.logits - gmmS.logits(1))));
        logphi = gmmS.logits - lse;
        logcomp= dlarray(zeros(Nloc, K, 'like', Uin));
        for kk = 1:K
            muk  = gmmS.mu(:,kk)';
            vark = exp(gmmS.logvar(:,kk))';
            diff = Uin - muk;
            mah  = sum(diff.^2 ./ vark, 2);
            logcomp(:,kk) = logphi(kk) - 0.5*(mah + Dloc*log(2*pi) + sum(gmmS.logvar(:,kk)));
        end
        mx   = max(logcomp, [], 2);
        logp = mx + log(sum(exp(logcomp - mx), 2) + eps);
    end

    function plist = params_to_list(p, Vloc, gcn_L_loc, V_all_loc, cfgL)
        plist = {};
        f = fieldnames(p.enc);
        for i=1:numel(f), plist{end+1} = p.enc.(f{i}); end
        f = fieldnames(p.dec);
        for i=1:numel(f), plist{end+1} = p.dec.(f{i}); end
        for ll=1:gcn_L_loc
            f = fieldnames(p.gcn_knn{ll});
            for i=1:numel(f), plist{end+1} = p.gcn_knn{ll}.(f{i}); end
        end
        for vv=1:Vloc
            for ll=1:gcn_L_loc
                f = fieldnames(p.gcn{vv}{ll});
                for i=1:numel(f), plist{end+1} = p.gcn{vv}{ll}.(f{i}); end
            end
        end
        for vv=1:V_all_loc
            f = fieldnames(p.sg_att{vv});
            for i=1:numel(f), plist{end+1} = p.sg_att{vv}.(f{i}); end
        end
        f = fieldnames(p.mg_att);
        for i=1:numel(f), plist{end+1} = p.mg_att.(f{i}); end
        if cfgL.use_gated_fusion && isfield(p,'gf')
            f = fieldnames(p.gf);
            for i=1:numel(f), plist{end+1} = p.gf.(f{i}); end
        end
        for b=1:numel(p.flow.blocks)
            blk = p.flow.blocks{b};
            if isfield(blk,'an') && cfgL.use_actnorm
                f = fieldnames(blk.an);
                for i=1:numel(f)
                    if isa(blk.an.(f{i}),'dlarray'), plist{end+1} = blk.an.(f{i}); end
                end
            end
            if isfield(blk,'mp')
                f = fieldnames(blk.mp);
                for i=1:numel(f)
                    if isa(blk.mp.(f{i}),'dlarray'), plist{end+1} = blk.mp.(f{i}); end
                end
            end
            if isfield(blk,'snet')
                f = fieldnames(blk.snet);
                for i=1:numel(f), plist{end+1} = blk.snet.(f{i}); end
            end
            if isfield(blk,'tnet')
                f = fieldnames(blk.tnet);
                for i=1:numel(f), plist{end+1} = blk.tnet.(f{i}); end
            end
        end
        plist{end+1} = p.gmm.logits;
        plist{end+1} = p.gmm.mu;
        plist{end+1} = p.gmm.logvar;
    end

    function [plist, pnames] = flatten_params(p, Vloc, gcn_L_loc, V_all_loc, cfgL)
        plist = {}; pnames = {};
        f = fieldnames(p.enc);
        for i=1:numel(f)
            plist{end+1}=p.enc.(f{i}); pnames{end+1}=['enc.' f{i}]; end
        f = fieldnames(p.dec);
        for i=1:numel(f)
            plist{end+1}=p.dec.(f{i}); pnames{end+1}=['dec.' f{i}]; end
        for ll=1:gcn_L_loc
            f = fieldnames(p.gcn_knn{ll});
            for i=1:numel(f)
                plist{end+1}=p.gcn_knn{ll}.(f{i});
                pnames{end+1}=sprintf('gcn_knn{%d}.%s',ll,f{i}); end
        end
        for vv=1:Vloc
            for ll=1:gcn_L_loc
                f = fieldnames(p.gcn{vv}{ll});
                for i=1:numel(f)
                    plist{end+1}=p.gcn{vv}{ll}.(f{i});
                    pnames{end+1}=sprintf('gcn{%d}{%d}.%s',vv,ll,f{i}); end
            end
        end
        for vv=1:V_all_loc
            f = fieldnames(p.sg_att{vv});
            for i=1:numel(f)
                plist{end+1}=p.sg_att{vv}.(f{i});
                pnames{end+1}=sprintf('sg_att{%d}.%s',vv,f{i}); end
        end
        f = fieldnames(p.mg_att);
        for i=1:numel(f)
            plist{end+1}=p.mg_att.(f{i}); pnames{end+1}=['mg_att.' f{i}]; end
        if cfgL.use_gated_fusion && isfield(p,'gf')
            f = fieldnames(p.gf);
            for i=1:numel(f)
                plist{end+1}=p.gf.(f{i}); pnames{end+1}=['gf.' f{i}]; end
        end
        for b=1:numel(p.flow.blocks)
            blk = p.flow.blocks{b};
            if isfield(blk,'an') && cfgL.use_actnorm
                f = fieldnames(blk.an);
                for i=1:numel(f)
                    if isa(blk.an.(f{i}),'dlarray')
                        plist{end+1}=blk.an.(f{i});
                        pnames{end+1}=sprintf('flow.blocks{%d}.an.%s',b,f{i}); end, end
            end
            if isfield(blk,'mp')
                f = fieldnames(blk.mp);
                for i=1:numel(f)
                    if isa(blk.mp.(f{i}),'dlarray')
                        plist{end+1}=blk.mp.(f{i});
                        pnames{end+1}=sprintf('flow.blocks{%d}.mp.%s',b,f{i}); end, end
            end
            if isfield(blk,'snet')
                f = fieldnames(blk.snet);
                for i=1:numel(f)
                    plist{end+1}=blk.snet.(f{i});
                    pnames{end+1}=sprintf('flow.blocks{%d}.snet.%s',b,f{i}); end
            end
            if isfield(blk,'tnet')
                f = fieldnames(blk.tnet);
                for i=1:numel(f)
                    plist{end+1}=blk.tnet.(f{i});
                    pnames{end+1}=sprintf('flow.blocks{%d}.tnet.%s',b,f{i}); end
            end
        end
        plist{end+1}=p.gmm.logits; pnames{end+1}='gmm.logits';
        plist{end+1}=p.gmm.mu;     pnames{end+1}='gmm.mu';
        plist{end+1}=p.gmm.logvar; pnames{end+1}='gmm.logvar';
    end

    function p = unflatten_params(plist, pnames, cfgL, Vloc, gcn_L_loc, V_all_loc)
        p         = struct();
        p.enc     = struct();
        p.dec     = struct();
        p.gcn_knn = cell(1, gcn_L_loc);
        for ll=1:gcn_L_loc, p.gcn_knn{ll}=struct(); end
        p.gcn     = cell(1, Vloc);
        for vv=1:Vloc
            p.gcn{vv} = cell(1, gcn_L_loc);
            for ll=1:gcn_L_loc, p.gcn{vv}{ll}=struct(); end
        end
        p.sg_att  = cell(1, V_all_loc);
        for vv=1:V_all_loc, p.sg_att{vv}=struct(); end
        p.mg_att  = struct();
        if cfgL.use_gated_fusion, p.gf = struct(); end
        p.flow    = struct();
        p.flow.blocks = {};
        p.gmm     = struct();

        for idx = 1:numel(plist)
            nm    = pnames{idx};
            if startsWith(nm,'enc.')
                p.enc.(extractAfter(nm,'enc.')) = plist{idx};
            elseif startsWith(nm,'dec.')
                p.dec.(extractAfter(nm,'dec.')) = plist{idx};
            elseif startsWith(nm,'mg_att.')
                p.mg_att.(extractAfter(nm,'mg_att.')) = plist{idx};
            elseif startsWith(nm,'gf.')
                p.gf.(extractAfter(nm,'gf.')) = plist{idx};
            elseif startsWith(nm,'gmm.')
                p.gmm.(extractAfter(nm,'gmm.')) = plist{idx};
            else
                t1 = regexp(nm,'^gcn_knn\{(\d+)\}\.(\w+)$','tokens','once');
                t2 = regexp(nm,'^gcn\{(\d+)\}\{(\d+)\}\.(\w+)$','tokens','once');
                t3 = regexp(nm,'^sg_att\{(\d+)\}\.(\w+)$','tokens','once');
                t4 = regexp(nm,'^flow\.blocks\{(\d+)\}\.(an|mp|snet|tnet)\.(\w+)$','tokens','once');
                if ~isempty(t1)
                    ll = str2double(t1{1});
                    p.gcn_knn{ll}.(t1{2}) = plist{idx};
                elseif ~isempty(t2)
                    vv = str2double(t2{1}); ll = str2double(t2{2});
                    p.gcn{vv}{ll}.(t2{3}) = plist{idx};
                elseif ~isempty(t3)
                    vv = str2double(t3{1});
                    p.sg_att{vv}.(t3{2}) = plist{idx};
                elseif ~isempty(t4)
                    b   = str2double(t4{1}); subf = t4{2}; fn = t4{3};
                    if numel(p.flow.blocks) < b
                        p.flow.blocks{b} = struct();
                    end
                    if ~isfield(p.flow.blocks{b}, subf)
                        p.flow.blocks{b}.(subf) = struct();
                    end
                    p.flow.blocks{b}.(subf).(fn) = plist{idx};
                end
            end
        end

        D_loc = cfgL.gcn_hidden;
        for b = 1:numel(p.flow.blocks)
            if ~isfield(p.flow.blocks{b},'mask')
                if mod(b,2)==1
                    p.flow.blocks{b}.mask = [ones(1,ceil(D_loc/2)),  zeros(1,floor(D_loc/2))];
                else
                    p.flow.blocks{b}.mask = [zeros(1,ceil(D_loc/2)), ones(1,floor(D_loc/2))];
                end
            end
            if ~isfield(p.flow.blocks{b},'mp'), p.flow.blocks{b}.mp = struct(); end
            if ~isfield(p.flow.blocks{b}.mp,'num_rel')
                p.flow.blocks{b}.mp.num_rel = cfgL.num_rel_total;
            end
        end
        if ~isfield(p.gmm,'K'),       p.gmm.K       = cfgL.gmm.K;    end
        if ~isfield(p.gmm,'diagCov'), p.gmm.diagCov = true;           end
    end

    function s = safe_silhouette(Emb, lbl)
        try
            sv = silhouette(Emb, lbl);
            sv = sv(~isnan(sv));
            if isempty(sv), s = 0; else, s = mean(sv); end
        catch
            s = 0;
        end
    end

    function [MQs, n_active, n_zero_e] = compute_MQ(A_bin, clusters)
        A_bin   = double(A_bin > 0);
        uq      = unique(clusters);
        K       = numel(uq);
        MQk     = nan(K, 1);
        for kk = 1:K
            idx   = find(clusters == uq(kk));
            other = find(clusters ~= uq(kk));
            if isempty(idx), continue; end
            I = sum(A_bin(idx, idx),   'all');
            E = sum(A_bin(idx, other), 'all');
            if (I + E) > 0
                MQk(kk) = I / (I + E);
            end
        end
        active   = ~isnan(MQk);
        n_active = sum(active);
        n_zero_e = K - n_active;
        if n_active > 0
            MQs = mean(MQk(active));
        else
            MQs = 0;
        end
    end

    function [MQh, n_active, n_zero_e] = compute_MQ_hybrid(A_bin, Z, clusters)
        A_bin = double(A_bin > 0);
        Znorm = Z ./ (sqrt(sum(Z.^2, 2)) + eps);
        Sim   = max(Znorm * Znorm', 0);
        uq    = unique(clusters);
        K     = numel(uq);
        MQk   = nan(K, 1);
        for kk = 1:K
            idx   = find(clusters == uq(kk));
            other = find(clusters ~= uq(kk));
            if isempty(idx), continue; end
            I = sum(A_bin(idx, idx)   .* Sim(idx, idx),   'all');
            E = sum(A_bin(idx, other) .* Sim(idx, other), 'all');
            if (I + E) > 0
                MQk(kk) = I / (I + E);
            end
        end
        active   = ~isnan(MQk);
        n_active = sum(active);
        n_zero_e = K - n_active;
        if n_active > 0
            MQh = mean(MQk(active));
        else
            MQh = 0;
        end
    end

    function nmi = computeNMI(tl, pl)
        tl = double(tl(:)); pl = double(pl(:)); n = numel(tl);
        cl = unique(tl); ck = unique(pl);
        nc = numel(cl);  nk = numel(ck);
        C  = zeros(nc, nk);
        for i=1:nc, for j=1:nk, C(i,j)=sum(tl==cl(i)&pl==ck(j)); end, end
        Ni = sum(C,2); Nj = sum(C,1); MI = 0;
        for i=1:nc, for j=1:nk
            if C(i,j)>0, MI=MI+(C(i,j)/n)*log(C(i,j)*n/(Ni(i)*Nj(j))); end
        end, end
        Hi  = -sum((Ni/n) .* log(Ni/n + eps));
        Hj  = -sum((Nj/n) .* log(Nj/n + eps));
        nmi = max(0, MI / (sqrt(Hi*Hj) + eps));
    end

    function ari = computeARI(tl, pl)
        tl = double(tl(:)); pl = double(pl(:)); n = numel(tl);
        cl = unique(tl); ck = unique(pl);
        nc = numel(cl);  nk = numel(ck);
        C  = zeros(nc, nk);
        for i=1:nc, for j=1:nk, C(i,j)=sum(tl==cl(i)&pl==ck(j)); end, end
        sc   = sum(C(:) .* (C(:)-1) / 2);
        rs   = sum(C,2); cs = sum(C,1);
        sca  = sum(rs .* (rs-1) / 2);
        scb  = sum(cs .* (cs-1) / 2);
        exp_ = (sca * scb) / (n*(n-1)/2);
        mx   = (sca + scb) / 2;
        if abs(mx - exp_) < eps, ari = 1; else, ari = (sc-exp_)/(mx-exp_); end
    end

    function Aknn = build_knn(Xmat, K)
        sim  = 1 - pdist2(Xmat, Xmat, 'cosine');
        Nloc = size(sim, 1);
        A    = zeros(Nloc);
        for i = 1:Nloc
            row    = sim(i, :);
            row(i) = -inf;
            [~, idx] = maxk(row, K);
            A(i, idx) = 1;
        end
        Aknn = double(A | A');
    end

    function M = init_glorot(r, c)
        lim = sqrt(6 / (r + c));
        M   = (rand(r,c) * 2 - 1) * lim;
    end

    function y = relu(x),    y = max(x, 0);          end
    function y = sigmoid(x), y = 1 ./ (1 + exp(-x)); end

    function write_log_header(fid, cfgL)
        fprintf(fid,'================================================================\n');
        fprintf(fid,' MGCCN-GNF v5\n');
        fprintf(fid,' Ref: Wang et al., IET Signal Processing 16(6):650-661, 2022\n');
        fprintf(fid,' Domain: Software Module Clustering\n');
        fprintf(fid,' Date: %s\n', datestr(now));
        fprintf(fid,'================================================================\n\n');
        fprintf(fid,'--- Model Configuration ---\n');
        fprintf(fid,'  Encoder       : [%s -> %d]\n', num2str(cfgL.encoder_dims), cfgL.latent_dim);
        fprintf(fid,'  gcn_hidden    : %d\n', cfgL.gcn_hidden);
        fprintf(fid,'  gcn_layers    : %d\n', cfgL.gcn_layers);
        fprintf(fid,'  epsilon       : %.3f\n', cfgL.epsilon);
        fprintf(fid,'  use_gated_fusion: %d\n', cfgL.use_gated_fusion);
        fprintf(fid,'  use_actnorm   : %d\n', cfgL.use_actnorm);
        fprintf(fid,'  alpha / beta  : %.3f / %.6f\n', cfgL.alpha, cfgL.beta);
        fprintf(fid,'--- Flow Configuration ---\n');
        fprintf(fid,'  numBlocks     : %d\n', cfgL.flow.numBlocks);
        fprintf(fid,'  lambda_flow   : %.4f\n', cfgL.flow.lambda);
        fprintf(fid,'  freeze_epochs : %d\n', cfgL.flow.freeze_epochs);
        fprintf(fid,'  struct_lambda : %.4f\n', cfgL.flow.struct_lambda);
        fprintf(fid,'  gcl_lambda    : %.4f\n', cfgL.flow.gcl_lambda);
        fprintf(fid,'  modbal_lambda : %.4f\n', cfgL.flow.modbal_lambda);
        fprintf(fid,'--- Training Configuration ---\n');
        fprintf(fid,'  maxEpochs     : %d\n', cfgL.maxEpochs);
        fprintf(fid,'  lr            : %.5f\n', cfgL.lr);
        fprintf(fid,'  k_method      : %s\n', cfgL.k_selection_method);
        fprintf(fid,'  seeds         : %d - %d\n\n', cfgL.seed_start, cfgL.seed_end);
    end

    function make_publication_figures(Emb, Z_fused, labels_pred, labels, ...
            lossHistory, lossComp, A_union, A_per_rel, Atilde, cfgL, seed, ...
            silVal, MQ_std, MQ_hyb, nmiVal, ariVal, best_k, MQ_per_rel)

        fig_dir = fullfile(cfgL.outdir, sprintf('figures_seed%d', seed));
        if ~exist(fig_dir,'dir'), mkdir(fig_dir); end
        has_gt   = ~isempty(labels);
        rel_names= cfgL.rel_names;
        V_rel    = numel(rel_names);
        ep       = 1:numel(lossHistory);

        try
            fig1 = figure('Visible','off','Position',[100 100 1000 420]);
            subplot(1,2,1);
            plot(ep, lossHistory, 'b-', 'LineWidth', 1.8);
            xlabel('Epoch'); ylabel('Total Loss');
            title('Training Loss', 'FontWeight','bold'); grid on;

            subplot(1,2,2);
            comp_nm = {'L_{res}','L_m','L_{flow}','L_{struct}','L_{gcl}','L_{modbal}'};
            cmap    = lines(6);
            hold on;
            for ci = 1:6
                if any(lossComp(:,ci) ~= 0)
                    semilogy(ep, abs(lossComp(:,ci))+1e-12, ...
                        'Color',cmap(ci,:),'LineWidth',1.3,'DisplayName',comp_nm{ci});
                end
            end
            hold off; legend('Location','northeast','FontSize',9);
            xlabel('Epoch'); ylabel('Loss (log)');
            title('Loss Components', 'FontWeight','bold'); grid on;
            saveas(fig1, fullfile(fig_dir,'fig01_loss_curve.png')); close(fig1);
        catch e, fprintf('  [VIS] Fig1 failed: %s\n',e.message); end

        try
            Zts = Emb;
            if size(Zts,2) > 50, [~,sc]=pca(Zts); Zts=sc(:,1:50); end
            perp = min(30, floor(size(Emb,1)/4));
            Y2   = tsne(Zts, 'NumDimensions',2, 'Perplexity',perp);

            n_sub = 1 + has_gt;
            fig2  = figure('Visible','off','Position',[100 100 550*n_sub 460]);
            subplot(1, n_sub, 1);
            scatter(Y2(:,1), Y2(:,2), 35, labels_pred, 'filled');
            colormap(lines(best_k)); colorbar; axis equal tight;
            title(sprintf('t-SNE — Prediction (k=%d)', best_k), 'FontWeight','bold');
            xlabel('t-SNE 1'); ylabel('t-SNE 2');

            if has_gt
                subplot(1,2,2);
                scatter(Y2(:,1), Y2(:,2), 35, labels, 'filled');
                colormap(lines(numel(unique(labels)))); colorbar; axis equal tight;
                title('t-SNE — Ground Truth', 'FontWeight','bold');
                xlabel('t-SNE 1'); ylabel('t-SNE 2');
            end
            saveas(fig2, fullfile(fig_dir,'fig02_tsne.png')); close(fig2);
        catch e, fprintf('  [VIS] Fig2 failed: %s\n',e.message); end

        try
            sv    = silhouette(Emb, labels_pred);
            uq    = unique(labels_pred);
            K_c   = numel(uq);
            sil_k = arrayfun(@(k)mean(sv(labels_pred==k)), uq);
            sz_k  = arrayfun(@(k)sum(labels_pred==k), uq);
            [~,ord] = sort(sil_k,'descend');

            fig3 = figure('Visible','off','Position',[100 100 max(600,K_c*60) 420]);
            bh   = bar(sil_k(ord), 'FaceColor','flat');
            bh.CData = cool(K_c);
            hold on;
            yline(silVal, 'r--', 'LineWidth',1.5, 'Label',sprintf('Mean=%.3f',silVal));
            hold off;
            xticks(1:K_c);
            xticklabels(arrayfun(@(x)sprintf('C%d\n(n=%d)',uq(ord(x)),sz_k(ord(x))), ...
                        1:K_c,'UniformOutput',false));
            xlabel('Cluster'); ylabel('Mean Silhouette');
            title('Silhouette per Cluster','FontWeight','bold'); grid on;
            saveas(fig3, fullfile(fig_dir,'fig03_silhouette_bar.png')); close(fig3);
        catch e, fprintf('  [VIS] Fig3 failed: %s\n',e.message); end

        try
            fig4 = figure('Visible','off','Position',[100 100 700 420]);
            mq_vals = [MQ_std, MQ_hyb, MQ_per_rel];
            mq_nms  = [{'MQ Union','MQ Hybrid'}, rel_names];
            cmap4   = [0.3 0.6 0.9; 0.9 0.4 0.2; ...
                       repmat([0.4 0.75 0.4],V_rel,1)];
            b4 = bar(mq_vals, 'FaceColor','flat');
            for ci=1:numel(mq_vals), b4.CData(ci,:) = cmap4(min(ci,end),:); end
            xticks(1:numel(mq_nms)); xticklabels(mq_nms);
            ylabel('MQ Value'); ylim([0 1]);
            title('Modularization Quality per Relation','FontWeight','bold'); grid on;
            [~, idx_d] = max(MQ_per_rel);
            hold on;
            bar_pos = 2 + idx_d;
            plot(bar_pos, MQ_per_rel(idx_d)+0.04, 'rv', 'MarkerSize',12, ...
                'MarkerFaceColor','r', 'DisplayName','Dominant');
            hold off;
            legend('show','Location','northwest');
            saveas(fig4, fullfile(fig_dir,'fig04_mq_per_relasi.png')); close(fig4);
        catch e, fprintf('  [VIS] Fig4 failed: %s\n',e.message); end

        try
            uq  = unique(labels_pred);
            K_c = numel(uq);
            mq_mat = zeros(K_c, V_rel);
            for ki = 1:K_c
                for vi = 1:V_rel
                    Ab   = A_per_rel{vi};
                    idx  = find(labels_pred == uq(ki));
                    oth  = find(labels_pred ~= uq(ki));
                    I    = sum(Ab(idx,idx),'all');
                    E    = sum(Ab(idx,oth),'all');
                    if (I+E)>0, mq_mat(ki,vi) = I/(I+E); end
                end
            end
            fig5 = figure('Visible','off','Position',[100 100 500 max(350,K_c*30)]);
            imagesc(mq_mat, [0 1]); colormap(hot); colorbar;
            xticks(1:V_rel); xticklabels(rel_names);
            yticks(1:K_c);  yticklabels(arrayfun(@(x)sprintf('C%d',x),uq,'UniformOutput',false));
            xlabel('Relation Type'); ylabel('Cluster');
            title('MQ per Cluster per Relation','FontWeight','bold');
            for i=1:K_c, for j=1:V_rel
                text(j,i,sprintf('%.2f',mq_mat(i,j)), ...
                    'HorizontalAlignment','center','Color','w','FontSize',8);
            end, end
            saveas(fig5, fullfile(fig_dir,'fig05_mq_heatmap.png')); close(fig5);
        catch e, fprintf('  [VIS] Fig5 failed: %s\n',e.message); end

        if has_gt
            try
                cl_t = unique(labels);
                cl_p = unique(labels_pred);
                CM   = zeros(numel(cl_t), numel(cl_p));
                for i=1:numel(cl_t), for j=1:numel(cl_p)
                    CM(i,j) = sum(labels==cl_t(i) & labels_pred==cl_p(j));
                end, end
                fig6 = figure('Visible','off','Position',[100 100 600 520]);
                imagesc(CM); colormap(hot); colorbar;
                xlabel('Prediction'); ylabel('Ground Truth');
                title(sprintf('Confusion Matrix  NMI=%.4f  ARI=%.4f',nmiVal,ariVal), ...
                    'FontWeight','bold');
                xticks(1:numel(cl_p)); xticklabels(cl_p);
                yticks(1:numel(cl_t)); yticklabels(cl_t);
                for i=1:size(CM,1), for j=1:size(CM,2)
                    text(j,i,num2str(CM(i,j)),'HorizontalAlignment','center','Color','w','FontSize',7);
                end, end
                saveas(fig6, fullfile(fig_dir,'fig06_confusion_matrix.png')); close(fig6);
            catch e, fprintf('  [VIS] Fig6 failed: %s\n',e.message); end
        end

        try
            uq  = unique(labels_pred);
            K_c = numel(uq);
            n_v = V_rel + 1;
            rel_nm_full = ['KNN', rel_names];
            fig7 = figure('Visible','off','Position',[100 100 320*n_v 300]);
            for vv = 1:n_v
                if vv == 1, Av_b = double(Atilde{1} > 0.01);
                else,        Av_b = A_per_rel{vv-1}; end
                dm = zeros(K_c, K_c);
                for i=1:K_c, for j=1:K_c
                    ii = find(labels_pred==uq(i)); jj=find(labels_pred==uq(j));
                    pp = numel(ii)*numel(jj);
                    if pp>0, dm(i,j)=sum(Av_b(ii,jj),'all')/pp; end
                end, end
                subplot(1,n_v,vv);
                imagesc(dm,[0 1]); colormap(hot);
                title(rel_nm_full{vv},'FontWeight','bold','FontSize',9);
                xlabel('Cluster'); ylabel('Cluster');
                if vv==n_v, colorbar; end
                xticks(1:K_c); yticks(1:K_c);
            end
            sgtitle('Edge Density between Clusters per Relation','FontWeight','bold');
            saveas(fig7, fullfile(fig_dir,'fig07_graph_density.png')); close(fig7);
        catch e, fprintf('  [VIS] Fig7 failed: %s\n',e.message); end

        try
            [~,~,eigv] = pca(Emb);
            eigv_pct   = eigv / sum(eigv) * 100;
            cum_v      = cumsum(eigv_pct);
            n_show     = min(20, numel(eigv_pct));
            fig8 = figure('Visible','off','Position',[100 100 600 380]);
            yyaxis left;
            bar(eigv_pct(1:n_show), 'FaceColor',[0.3 0.6 0.9]);
            ylabel('Variance Explained (%)');
            yyaxis right;
            plot(cum_v(1:n_show), 'r-o', 'LineWidth',1.5, 'MarkerSize',5);
            yline(90,'k--','90%','LineWidth',1);
            ylabel('Cumulative (%)');
            xlabel('Principal Component');
            title('PCA Scree (Embedding Space)','FontWeight','bold'); grid on;
            saveas(fig8, fullfile(fig_dir,'fig08_pca_scree.png')); close(fig8);
        catch e, fprintf('  [VIS] Fig8 failed: %s\n',e.message); end

        try
            uq  = unique(labels_pred);
            szk = arrayfun(@(k)sum(labels_pred==k), uq);
            fig9= figure('Visible','off','Position',[100 100 max(600,K_c*60) 380]);
            bar(szk, 'FaceColor',[0.4 0.75 0.4]);
            hold on;
            yline(mean(szk),'r--','LineWidth',1.5,'Label',sprintf('Mean=%d',round(mean(szk))));
            hold off;
            xticks(1:numel(uq));
            xticklabels(arrayfun(@(x)sprintf('C%d',x),uq,'UniformOutput',false));
            xlabel('Cluster'); ylabel('Number of Nodes');
            title(sprintf('Cluster Sizes (k=%d)',best_k),'FontWeight','bold'); grid on;
            saveas(fig9, fullfile(fig_dir,'fig09_cluster_size.png')); close(fig9);
        catch e, fprintf('  [VIS] Fig9 failed: %s\n',e.message); end

        try
            mv = [silVal, MQ_std, MQ_hyb, MQ_per_rel];
            mn = [{'Sil','MQ-Std','MQ-Hyb'}, rel_names];
            if has_gt
                mv = [mv, nmiVal, max(0,ariVal)];
                mn = [mn, {'NMI','ARI'}];
            end
            n_m   = numel(mv);
            theta = linspace(0, 2*pi, n_m+1); theta(end) = [];
            fig10 = figure('Visible','off','Position',[100 100 520 520]);
            ax    = polaraxes;
            r_val = [mv, mv(1)];
            th_val= [theta, theta(1)];
            polarplot(ax, th_val, r_val, 'b-o', 'LineWidth',2, 'MarkerFaceColor','b');
            ax.ThetaTick     = rad2deg(theta);
            ax.ThetaTickLabel= mn;
            ax.RLim = [0 1]; ax.FontSize = 9;
            title(sprintf('Metric Summary — Seed %d', seed), ...
                'FontWeight','bold','Units','normalized','Position',[0.5 1.03 0]);
            saveas(fig10, fullfile(fig_dir,'fig10_radar_metrics.png')); close(fig10);
        catch e, fprintf('  [VIS] Fig10 failed: %s\n',e.message); end

        fprintf('  [VIS] Figures saved to: %s\n', fig_dir);
    end

end