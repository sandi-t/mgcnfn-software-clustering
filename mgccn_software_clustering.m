function mgccn_software_clustering()
clc; close all;

cfg = struct();

cfg.A_paths = {
    './javacc/adjacency_matrix_aggregation.txt', ...
    './javacc/adjacency_matrix_association.txt', ...
    './javacc/adjacency_matrix_composition.txt', ...
    './javacc/adjacency_matrix_dependency.txt'
};
cfg.features = './javacc/node_features.txt';
cfg.labels   = './javacc/label.txt';

cfg.num_packages = 4;
cfg.k_selection = 'fixed';

cfg.latent_dim   = 64;
cfg.gcn_hidden   = 64;
cfg.encoder_dims = [512, 256];
cfg.gcn_layers   = 2;
cfg.att_h        = 64;
cfg.epsilon      = 0.5;
cfg.K            = 4;

cfg.alpha = 1.0;
cfg.beta  = 1e-4;

cfg.maxEpochs    = 300;
cfg.lr           = 1e-3;
cfg.verbose      = true;
cfg.log_interval = 50;

cfg.seed_start = 1;
cfg.seed_end   = 2;

cfg.outdir = fullfile(pwd, 'mgccn_results');
if ~exist(cfg.outdir, 'dir'), mkdir(cfg.outdir); end
logfile = fullfile(cfg.outdir, sprintf('mgccn_log_%s.txt', datestr(now,'yyyymmdd_HHMMSS')));
flog = fopen(logfile, 'w');

V = numel(cfg.A_paths);
graphs = cell(1,V);
for v = 1:V
    A = double(load(cfg.A_paths{v}));
    A = (A + A')/2;
    graphs{v} = A;
end
X = double(load(cfg.features));
[N, D] = size(X);
has_gt = false;
gt_labels = [];
if isfile(cfg.labels)
    gt_labels = load(cfg.labels);
    gt_labels = gt_labels(:);
    if length(gt_labels) == N
        has_gt = true;
        num_classes = numel(unique(gt_labels));
        fprintf(flog, 'Ground truth labels loaded: %d classes\n', num_classes);
    else
        warning('Label count mismatch');
    end
end

fprintf('Data loaded: N=%d, D=%d, V=%d\n', N, D, V);
fprintf(flog, 'Data loaded: N=%d, D=%d, V=%d\n', N, D, V);

A_knn = build_knn(X, cfg.K);
graphs_all = [{A_knn}, graphs];
V_all = numel(graphs_all);

Atilde = cell(1, V_all);
for v = 1:V_all
    Av = (graphs_all{v} + graphs_all{v}')/2;
    Av = Av + eye(N);
    deg = sum(Av,2);
    Dm = diag(1 ./ sqrt(max(deg,eps)));
    Atilde{v} = Dm * Av * Dm;
end

A_union = zeros(N,N);
for v = 1:V_all
    A_union = A_union + graphs_all{v};
end
A_union = double(A_union > 0);

enc_sizes = [D, cfg.encoder_dims, cfg.latent_dim];
Lenc = numel(enc_sizes) - 1;
dec_sizes = [cfg.latent_dim, fliplr(cfg.encoder_dims), D];
Ldec = numel(dec_sizes) - 1;
d = cfg.gcn_hidden;

seeds = cfg.seed_start : cfg.seed_end;
nSeeds = numel(seeds);
all_results = zeros(nSeeds, 12);
result_labels = cell(nSeeds,1);

for si = 1:nSeeds
    seed = seeds(si);
    rng(seed, 'twister');
    fprintf('\n===== Seed %d/%d =====\n', seed, nSeeds);
    fprintf(flog, '\n===== Seed %d/%d =====\n', seed, nSeeds);

    params = struct();
    params.enc = struct();
    for l = 1:Lenc
        params.enc.(['W' num2str(l)]) = dlarray(init_glorot(enc_sizes(l+1), enc_sizes(l)));
        params.enc.(['b' num2str(l)]) = dlarray(zeros(enc_sizes(l+1),1));
    end
    params.dec = struct();
    for l = 1:Ldec
        params.dec.(['W' num2str(l)]) = dlarray(init_glorot(dec_sizes(l+1), dec_sizes(l)));
        params.dec.(['b' num2str(l)]) = dlarray(zeros(dec_sizes(l+1),1));
    end
    params.gcn_knn = cell(1, cfg.gcn_layers);
    for l = 1:cfg.gcn_layers
        params.gcn_knn{l}.W = dlarray(init_glorot(d,d));
        params.gcn_knn{l}.b = dlarray(zeros(d,1));
    end
    params.gcn = cell(1,V);
    for v = 1:V
        params.gcn{v} = cell(1, cfg.gcn_layers);
        for l = 1:cfg.gcn_layers
            inS = D*(l==1) + d*(l>1);
            params.gcn{v}{l}.W = dlarray(init_glorot(d,inS));
            params.gcn{v}{l}.b = dlarray(zeros(d,1));
        end
    end
    params.sg_att = cell(1, V_all);
    for v = 1:V_all
        params.sg_att{v}.w = dlarray(init_glorot(d,1)');
        params.sg_att{v}.b = dlarray(0);
    end
    params.mg_att.W = dlarray(init_glorot(cfg.att_h, d));
    params.mg_att.b = dlarray(zeros(cfg.att_h,1));
    params.mg_att.q = dlarray(init_glorot(cfg.att_h,1));

    [paramList, paramNames] = flatten_params(params, V, cfg.gcn_layers, V_all);
    numP = numel(paramList);
    m_ad = cell(1,numP); v_ad = cell(1,numP);
    for i = 1:numP
        if isa(paramList{i}, 'dlarray')
            sz = size(extractdata(paramList{i}));
        else
            sz = size(paramList{i});
        end
        m_ad{i} = zeros(sz);
        v_ad{i} = zeros(sz);
    end
    Xdl = dlarray(X);
    iter = 0;
    cur_lr = cfg.lr;
    lossHistory = zeros(cfg.maxEpochs,1);

    for epoch = 1:cfg.maxEpochs
        [lossVal, grads, ~] = dlfeval(@(p) model_gradients(p, Atilde, Xdl, cfg, N, D, V, V_all, cfg.gcn_layers, d, Lenc), params);
        lossNum = double(gather(extractdata(lossVal)));
        lossHistory(epoch) = lossNum;

        for gi = 1:numP
            if ~isempty(grads{gi})
                grads{gi} = max(min(grads{gi}, 1.0), -1.0);
            end
        end
        iter = iter+1;
        for p = 1:numP
            g = grads{p};
            if isempty(g), continue; end
            if ~isa(g,'dlarray'), g = dlarray(double(g)); end
            gd = double(extractdata(g));
            m_ad{p} = 0.9*m_ad{p} + 0.1*gd;
            v_ad{p} = 0.999*v_ad{p} + 0.001*gd.^2;
            mhat = m_ad{p} / (1 - 0.9^iter);
            vhat = v_ad{p} / (1 - 0.999^iter);
            step = cur_lr * (mhat ./ (sqrt(vhat) + 1e-8));
            if isa(paramList{p}, 'dlarray')
                paramList{p} = dlarray(double(extractdata(paramList{p})) - step);
            else
                paramList{p} = dlarray(double(paramList{p}) - step);
            end
        end
        params = unflatten_params(paramList, paramNames, V, cfg.gcn_layers, V_all);
        if mod(epoch,50)==0
            cur_lr = cur_lr * 0.5;
        end
        if cfg.verbose && (mod(epoch,cfg.log_interval)==0 || epoch==1)
            fprintf('  Ep %4d/%d Loss=%.5e\n', epoch, cfg.maxEpochs, lossNum);
            fprintf(flog, '  Ep %4d/%d Loss=%.5e\n', epoch, cfg.maxEpochs, lossNum);
        end
    end

    [~, ~, outFinal] = dlfeval(@(p) model_gradients(p, Atilde, Xdl, cfg, N, D, V, V_all, cfg.gcn_layers, d, Lenc), params);
    Z_fused = double(gather(extractdata(outFinal.Z_fused)));

    if strcmp(cfg.k_selection, 'silhouette')
        best_k = cfg.num_packages;
        labels_pred = kmeans(Z_fused, best_k, 'Replicates', 20, 'MaxIter', 500, 'Start', 'plus');
    else
        Kmin = 3;
        Kmax = min(20, N-1);
        best_score = -inf;
        best_k = Kmin;
        for ktest = Kmin:Kmax
            lbl = kmeans(Z_fused, ktest, 'Replicates', 5, 'MaxIter', 300);
            sil = mean(silhouette(Z_fused, lbl));
            if sil > best_score
                best_score = sil;
                best_k = ktest;
            end
        end
        labels_pred = kmeans(Z_fused, best_k, 'Replicates', 20, 'MaxIter', 500);
    end
    result_labels{si} = labels_pred;

    sil = mean(silhouette(Z_fused, labels_pred));
    try, ev = evalclusters(Z_fused, labels_pred, 'CalinskiHarabasz'); ch = ev.CriterionValues; catch, ch = NaN; end
    try, ev = evalclusters(Z_fused, labels_pred, 'DaviesBouldin'); db = ev.CriterionValues; catch, db = NaN; end
    mq = compute_MQ_standard(A_union, labels_pred);

    if has_gt
        nmi = computeNMI(gt_labels, labels_pred);
        ari = computeARI(gt_labels, labels_pred);
        mojofm = compute_MoJoFM(gt_labels, labels_pred);
        turbomq = compute_TurboMQ(A_union, labels_pred);
        [coh, coup] = compute_cohesion_coupling(A_union, labels_pred);
    else
        nmi = NaN; ari = NaN; mojofm = NaN; turbomq = NaN; coh = NaN; coup = NaN;
    end

    all_results(si,:) = [seed, best_k, sil, ch, db, mq, mojofm, turbomq, coh, coup, nmi, ari];

    fprintf('  k=%d | Sil=%.4f | CH=%.2f | DB=%.4f | MQ=%.4f | NMI=%.4f | ARI=%.4f | MoJoFM=%.4f | TurboMQ=%.4f | Cohesion=%.4f | Coupling=%.4f\n', best_k, sil, ch, db, mq, nmi, ari, mojofm, turbomq, coh, coup);
    fprintf(flog, '  k=%d | Sil=%.4f | CH=%.2f | DB=%.4f | MQ=%.4f | NMI=%.4f | ARI=%.4f | MoJoFM=%.4f | TurboMQ=%.4f | Cohesion=%.4f | Coupling=%.4f\n', best_k, sil, ch, db, mq, nmi, ari, mojofm, turbomq, coh, coup);
end

fprintf('\n========== SUMMARY over %d seeds ==========\n', nSeeds);
fprintf(flog, '\n========== SUMMARY over %d seeds ==========\n', nSeeds);
metric_names = {'Seed','k','Silhouette','CH','DB','MQ','MoJoFM','TurboMQ','Cohesion','Coupling','NMI','ARI'};
for m = 3:12
    vals = all_results(:,m);
    if any(strcmp(metric_names{m}, {'DB','Coupling'}))
        best = min(vals);
    else
        best = max(vals);
    end
    fprintf('%-12s  mean=%.4f  std=%.4f  best=%.4f\n', metric_names{m}, mean(vals), std(vals), best);
    fprintf(flog, '%-12s  mean=%.4f  std=%.4f  best=%.4f\n', metric_names{m}, mean(vals), std(vals), best);
end

outfile = fullfile(cfg.outdir, 'mgccn_results.mat');
save(outfile, 'all_results', 'metric_names', 'result_labels', 'cfg');
csvfile = fullfile(cfg.outdir, 'mgccn_results.csv');
writetable(array2table(all_results, 'VariableNames', metric_names), csvfile);
fprintf('\nResults saved to %s and %s\n', outfile, csvfile);
fclose(flog);
fprintf('Log saved to %s\n', logfile);
end

function Aknn = build_knn(X, K)
sim = 1 - pdist2(X, X, 'cosine');
N = size(sim,1);
A = zeros(N);
for i = 1:N
    row = sim(i,:);
    row(i) = -inf;
    [~, idx] = maxk(row, K);
    A(i, idx) = 1;
end
Aknn = double(A | A');
end

function M = init_glorot(r,c)
lim = sqrt(6/(r+c));
M = (rand(r,c)*2-1)*lim;
end

function y = relu(x), y = max(0,x); end
function y = sigmoid(x), y = 1./(1+exp(-x)); end

function [loss, grads, outputs] = model_gradients(params, Atilde, Xdl, cfg, N, D, V, V_all, gcn_L, d, Lenc)
H = Xdl;
for ll = 1:Lenc
    W = params.enc.(['W' num2str(ll)]);
    b = params.enc.(['b' num2str(ll)]);
    H = H * W' + b';
    if ll < Lenc, H = relu(H); end
end
He = H;

Xhat = He;
Ldec = numel(fieldnames(params.dec))/2;
for ll = 1:Ldec
    Wd = params.dec.(['W' num2str(ll)]);
    bd = params.dec.(['b' num2str(ll)]);
    Xhat = Xhat * Wd' + bd';
    if ll < Ldec, Xhat = relu(Xhat); end
end
Xrec = Xhat;
L_res = (1/(2*N)) * sum((Xdl - Xrec).^2, 'all');

Z_knn = He;
Aknn_t = Atilde{1};
for ll = 1:gcn_L
    Z_tilde = (1-cfg.epsilon)*Z_knn + cfg.epsilon*He;
    Wk = params.gcn_knn{ll}.W;
    bk = params.gcn_knn{ll}.b;
    Z_knn = relu((Aknn_t * Z_tilde) * Wk' + bk');
end

Zv = cell(1, V);
for vv = 1:V
    Zcur = Xdl;
    At_v = Atilde{vv+1};
    for ll = 1:gcn_L
        Wg = params.gcn{vv}{ll}.W;
        bg = params.gcn{vv}{ll}.b;
        Zcur = relu((At_v * Zcur) * Wg' + bg');
    end
    Zv{vv} = Zcur;
end

Zhat = cell(1, V_all);
for vv = 1:V_all
    if vv == 1, Z_in = Z_knn; else, Z_in = Zv{vv-1}; end
    wv = params.sg_att{vv}.w;
    bv = params.sg_att{vv}.b;
    s = tanh(Z_in * wv' + bv);
    s_exp = exp(s - max(s,[],1));
    alpha = s_exp ./ (sum(s_exp,1)+eps);
    Zhat{vv} = sigmoid(alpha .* Z_in);
end

Wm = params.mg_att.W;
bm = params.mg_att.b;
qm = params.mg_att.q;
Omega = dlarray(zeros(N, V_all, 'like', Xdl));
for vv = 1:V_all
    H_v = tanh(Zhat{vv} * Wm' + bm');
    Omega(:,vv) = H_v * qm;
end
Ome_exp = exp(Omega - max(Omega,[],2));
Alpha_graph = Ome_exp ./ (sum(Ome_exp,2)+eps);
Z_fused = dlarray(zeros(N, d, 'like', Xdl));
for vv = 1:V_all
    Z_fused = Z_fused + Alpha_graph(:,vv) .* Zhat{vv};
end

Sv = cell(1, V_all);
for vv = 1:V_all
    Zt = Zhat{vv};
    rn = sqrt(sum(Zt.^2,2)+eps);
    Sv{vv} = (Zt./rn) * (Zt./rn)';
end
Sknn = Sv{1};
Lm = dlarray(0);
for vv = 2:V_all
    Lm = Lm + sum((Sknn - Sv{vv}).^2, 'all');
end
for i = 1:V_all
    for j = i+1:V_all
        Lm = Lm + 0.5 * sum((Sv{i} - Sv{j}).^2, 'all');
    end
end

loss = cfg.alpha * L_res + cfg.beta * Lm;
plist = params_to_list(params, V, gcn_L, V_all);
grads = dlgradient(loss, plist);
outputs.Z_fused = Z_fused;
outputs.Xrec = Xrec;
end

function plist = params_to_list(p, V, gcn_L, V_all)
plist = {};
f = fieldnames(p.enc); for i=1:numel(f), plist{end+1}=p.enc.(f{i}); end
f = fieldnames(p.dec); for i=1:numel(f), plist{end+1}=p.dec.(f{i}); end
for ll=1:gcn_L
    f = fieldnames(p.gcn_knn{ll}); for i=1:numel(f), plist{end+1}=p.gcn_knn{ll}.(f{i}); end
end
for vv=1:V
    for ll=1:gcn_L
        f = fieldnames(p.gcn{vv}{ll}); for i=1:numel(f), plist{end+1}=p.gcn{vv}{ll}.(f{i}); end
    end
end
for vv=1:V_all
    f = fieldnames(p.sg_att{vv}); for i=1:numel(f), plist{end+1}=p.sg_att{vv}.(f{i}); end
end
f = fieldnames(p.mg_att); for i=1:numel(f), plist{end+1}=p.mg_att.(f{i}); end
end

function [plist, pnames] = flatten_params(p, V, gcn_L, V_all)
plist = {}; pnames = {};
f = fieldnames(p.enc); for i=1:numel(f), plist{end+1}=p.enc.(f{i}); pnames{end+1}=['enc.' f{i}]; end
f = fieldnames(p.dec); for i=1:numel(f), plist{end+1}=p.dec.(f{i}); pnames{end+1}=['dec.' f{i}]; end
for ll=1:gcn_L
    f = fieldnames(p.gcn_knn{ll}); for i=1:numel(f), plist{end+1}=p.gcn_knn{ll}.(f{i}); pnames{end+1}=sprintf('gcn_knn{%d}.%s',ll,f{i}); end
end
for vv=1:V
    for ll=1:gcn_L
        f = fieldnames(p.gcn{vv}{ll}); for i=1:numel(f), plist{end+1}=p.gcn{vv}{ll}.(f{i}); pnames{end+1}=sprintf('gcn{%d}{%d}.%s',vv,ll,f{i}); end
    end
end
for vv=1:V_all
    f = fieldnames(p.sg_att{vv}); for i=1:numel(f), plist{end+1}=p.sg_att{vv}.(f{i}); pnames{end+1}=sprintf('sg_att{%d}.%s',vv,f{i}); end
end
f = fieldnames(p.mg_att); for i=1:numel(f), plist{end+1}=p.mg_att.(f{i}); pnames{end+1}=['mg_att.' f{i}]; end
end

function p = unflatten_params(plist, pnames, V, gcn_L, V_all)
p = struct();
p.enc = struct();
p.dec = struct();
p.gcn_knn = cell(1, gcn_L);
for ll = 1:gcn_L, p.gcn_knn{ll} = struct(); end
p.gcn = cell(1, V);
for vv = 1:V
    p.gcn{vv} = cell(1, gcn_L);
    for ll = 1:gcn_L, p.gcn{vv}{ll} = struct(); end
end
p.sg_att = cell(1, V_all);
for vv = 1:V_all, p.sg_att{vv} = struct(); end
p.mg_att = struct();

for idx = 1:numel(plist)
    nm = pnames{idx};
    if numel(nm) >= 4 && strcmp(nm(1:4), 'enc.')
        p.enc.(nm(5:end)) = plist{idx};
    elseif numel(nm) >= 4 && strcmp(nm(1:4), 'dec.')
        p.dec.(nm(5:end)) = plist{idx};
    elseif numel(nm) >= 7 && strcmp(nm(1:7), 'mg_att.')
        p.mg_att.(nm(8:end)) = plist{idx};
    else
        openBrace = strfind(nm, '{');
        closeBrace = strfind(nm, '}');
        dotPos = strfind(nm, '.');
        if isempty(openBrace) || isempty(closeBrace) || isempty(dotPos)
            continue;
        end
        if contains(nm, 'gcn_knn')
            idxStr = nm(openBrace(1)+1:closeBrace(1)-1);
            ll = str2double(idxStr);
            field = nm(dotPos(1)+1:end);
            if ll >= 1 && ll <= gcn_L
                p.gcn_knn{ll}.(field) = plist{idx};
            end
        elseif contains(nm, 'gcn{')
            idx1 = nm(openBrace(1)+1:closeBrace(1)-1);
            vv = str2double(idx1);
            if numel(openBrace) >= 2 && numel(closeBrace) >= 2
                idx2 = nm(openBrace(2)+1:closeBrace(2)-1);
                ll = str2double(idx2);
                field = nm(dotPos(1)+1:end);
                if vv >= 1 && vv <= V && ll >= 1 && ll <= gcn_L
                    p.gcn{vv}{ll}.(field) = plist{idx};
                end
            end
        elseif contains(nm, 'sg_att')
            idxStr = nm(openBrace(1)+1:closeBrace(1)-1);
            vv = str2double(idxStr);
            field = nm(dotPos(1)+1:end);
            if vv >= 1 && vv <= V_all
                p.sg_att{vv}.(field) = plist{idx};
            end
        end
    end
end
end

function mq = compute_MQ_standard(A_bin, clusters)
A_bin = double(A_bin > 0);
uq = unique(clusters);
K = numel(uq);
mqk = zeros(K,1);
for k = 1:K
    idx = find(clusters == uq(k));
    other = find(clusters ~= uq(k));
    I = sum(A_bin(idx,idx),'all');
    E = sum(A_bin(idx,other),'all');
    if I+E>0, mqk(k) = I/(I+E); end
end
mq = mean(mqk);
end

function mfm = compute_MoJoFM(true_labels, pred_labels)
true_labels = true_labels(:); pred_labels = pred_labels(:);
n = length(true_labels);
[~,~,true_idx] = unique(true_labels);
[~,~,pred_idx] = unique(pred_labels);
Kt = max(true_idx); Kp = max(pred_idx);
C = accumarray([true_idx, pred_idx], 1, [Kt, Kp]);
[vals, idx] = sort(C(:), 'descend');
matched = zeros(Kt,1);
used = false(1,Kp);
for i = 1:numel(vals)
    if vals(i)==0, break; end
    [r,c] = ind2sub([Kt, Kp], idx(i));
    if ~matched(r) && ~used(c)
        matched(r) = c;
        used(c) = true;
    end
end
moved = 0;
for t = 1:Kt
    sz = sum(true_idx==t);
    if matched(t) > 0
        moved = moved + (sz - C(t, matched(t)));
    else
        moved = moved + sz;
    end
end
mfm = 1 - moved/n;
end

function tmq = compute_TurboMQ(A_bin, clusters)
A_bin = double(A_bin>0);
uq = unique(clusters);
K = numel(uq);
mqk = zeros(K,1);
w = zeros(K,1);
for k = 1:K
    idx = find(clusters == uq(k));
    other = find(clusters ~= uq(k));
    I = sum(A_bin(idx,idx),'all');
    E = sum(A_bin(idx,other),'all');
    if I+E>0, mqk(k)=I/(I+E); end
    w(k) = length(idx);
end
tmq = sum(w.*mqk)/sum(w);
end

function [coh, coup] = compute_cohesion_coupling(A_bin, clusters)
A_bin = double(A_bin>0);
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
    I = sum(A_bin(idx,idx),'all');
    E = sum(A_bin(idx,other),'all');
    coh(k) = I / (n*(n-1));
    if I+E>0
        coup(k) = E/(I+E);
    else
        coup(k)=0;
    end
end
coh = mean(coh);
coup = mean(coup);
end

function nmi = computeNMI(true_labels, pred_labels)
true_labels = true_labels(:);
pred_labels = pred_labels(:);
n = length(true_labels);
[cl,~,true_idx] = unique(true_labels);
[ck,~,pred_idx] = unique(pred_labels);
nc = numel(cl); nk = numel(ck);
C = zeros(nc,nk);
for i=1:n, C(true_idx(i), pred_idx(i)) = C(true_idx(i), pred_idx(i)) + 1; end
Ni = sum(C,2); Nj = sum(C,1);
MI = 0;
for i=1:nc
    for j=1:nk
        if C(i,j)>0
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
[cl,~,true_idx] = unique(true_labels);
[ck,~,pred_idx] = unique(pred_labels);
nc = numel(cl); nk = numel(ck);
C = zeros(nc,nk);
for i=1:n, C(true_idx(i), pred_idx(i)) = C(true_idx(i), pred_idx(i)) + 1; end
sc = sum(C(:).*(C(:)-1)/2);
rs = sum(C,2); cs = sum(C,1);
sca = sum(rs.*(rs-1)/2);
scb = sum(cs.*(cs-1)/2);
exp_ = (sca * scb) / (n*(n-1)/2);
mx = (sca + scb)/2;
if abs(mx - exp_) < eps, ari = 1; else, ari = (sc - exp_) / (mx - exp_); end
end