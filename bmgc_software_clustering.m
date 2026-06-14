function results_table = bmgc_software_clustering(cfg)

if nargin < 1, cfg = struct(); end

if ~isfield(cfg,'A_paths') || isempty(cfg.A_paths)
    cfg.A_paths = {'./javacc/adjacency_matrix_aggregation.txt','./javacc/adjacency_matrix_association.txt',...
                   './javacc/adjacency_matrix_composition.txt','./javacc/adjacency_matrix_dependency.txt'};
end
if ~isfield(cfg,'features') || isempty(cfg.features)
    cfg.features = './javacc/node_features.txt';
end
if ~isfield(cfg,'labels') || isempty(cfg.labels)
    cfg.labels = './javacc/label.txt';
end

if ~isfield(cfg,'seed_start'),   cfg.seed_start   = 1;    end
if ~isfield(cfg,'seed_end'),     cfg.seed_end     = 50;   end
if ~isfield(cfg,'C'),            cfg.C            = 10;   end
if ~isfield(cfg,'epochs'),       cfg.epochs       = 300;  end
if ~isfield(cfg,'dr'),           cfg.dr           = 10;   end
if ~isfield(cfg,'K'),            cfg.K            = 3;    end
if ~isfield(cfg,'alpha'),        cfg.alpha        = 0.3;  end
if ~isfield(cfg,'tau'),          cfg.tau          = 1.0;  end
if ~isfield(cfg,'t_recalc'),     cfg.t_recalc     = 50;   end
if ~isfield(cfg,'lr'),           cfg.lr           = 1e-2; end
if ~isfield(cfg,'weight_decay'), cfg.weight_decay = 1e-4; end
if ~isfield(cfg,'kmeans_reps'),  cfg.kmeans_reps  = 10;   end
if ~isfield(cfg,'max_km_iter'),  cfg.max_km_iter  = 500;  end
if ~isfield(cfg,'save_csv'),     cfg.save_csv     = false; end
if ~isfield(cfg,'verbose'),      cfg.verbose      = true;  end
if ~isfield(cfg,'outdir'),       cfg.outdir       = 'bmgc_results'; end
if ~isfield(cfg,'k_selection_method'), cfg.k_selection_method = 'silhoutte'; end
if ~isfield(cfg,'k_range'),      cfg.k_range      = 3:20; end
if ~isfield(cfg,'k_fixed'),      cfg.k_fixed      = cfg.C; end

if ~exist(cfg.outdir, 'dir'), mkdir(cfg.outdir); end

seed_list = cfg.seed_start : cfg.seed_end;
n_seeds = numel(seed_list);

fprintf('=================================================================\n');
fprintf(' BMGC - Software Module Clustering (Baseline)\n');
fprintf(' Seeds  : %d to %d (%d runs)\n', cfg.seed_start, cfg.seed_end, n_seeds);
fprintf(' K selection: %s', cfg.k_selection_method);
if strcmp(cfg.k_selection_method, 'fixed')
    fprintf(' (C=%d)', cfg.k_fixed);
else
    fprintf(' (range %d:%d)', cfg.k_range(1), cfg.k_range(end));
end
fprintf(' | Epochs: %d | K: %d | alpha=%.2f | dr=%d | t_recalc=%d\n',...
        cfg.epochs, cfg.K, cfg.alpha, cfg.dr, cfg.t_recalc);
fprintf('=================================================================\n');

V = numel(cfg.A_paths);
A_list = cell(1,V);
for v = 1:V
    p = cfg.A_paths{v};
    if ~isfile(p), error('File not found: %s', p); end
    try; Atmp = readmatrix(p); catch; Atmp = dlmread(p); end
    Atmp = double(Atmp);
    Atmp = (Atmp+Atmp')/2;
    Atmp(Atmp<0) = 0;
    A_list{v} = sparse(Atmp);
end
N = size(A_list{1},1);

if ~isfile(cfg.features), error('Features not found: %s', cfg.features); end
try; X = readmatrix(cfg.features); catch; X = dlmread(cfg.features); end
X = double(X);
if size(X,1) ~= N, error('Feature rows (%d) != nodes (%d)', size(X,1), N); end
D = size(X,2);

has_gt = false;
gt_labels = [];
if ~isempty(cfg.labels) && isfile(cfg.labels)
    try; gt_labels = load(cfg.labels); catch; gt_labels = dlmread(cfg.labels); end
    gt_labels = gt_labels(:);
    if numel(gt_labels) == N
        has_gt = true;
        if cfg.verbose, fprintf('[Load] Ground truth loaded from %s\n', cfg.labels); end
    else
        warning('Label count (%d) != nodes (%d)', numel(gt_labels), N);
    end
end

if cfg.verbose
    fprintf('[Load] N=%d | D=%d | V=%d graphs\n', N, D, V);
end

A_hat_list = cell(1,V);
for v = 1:V
    A = full(A_list{v}) + eye(N);
    deg = sum(A,2); deg(deg==0)=1;
    Dinv = spdiags(deg.^(-0.5), 0, N, N);
    Ah = sparse(Dinv * A * Dinv);
    Ah(isnan(Ah)) = 0;
    A_hat_list{v} = Ah;
end

X(isnan(X)) = 0;
mu_X = mean(X,1); std_X = std(X,0,1); std_X(std_X==0)=1;
X = (X - mu_X) ./ std_X;
X = X ./ (sqrt(sum(X.^2,2)) + eps);

Xv_list = cell(1,V);
for v = 1:V
    Xv = X;
    for k = 1:cfg.K
        Xv = (1-cfg.alpha)*(A_hat_list{v}*Xv) + cfg.alpha*X;
    end
    Xv(isnan(Xv)) = 0;
    Xv_list{v} = Xv;
end

Z_init = cell(1,V);
for v=1:V, Z_init{v} = X; end
v_dom_init = find_dominant_view(X, Z_init, V, N);
if cfg.verbose, fprintf('[Init] Dominant view v* = %d\n\n', v_dom_init); end

A_union = sparse(N,N);
for v = 1:V, A_union = A_union + A_list{v}; end
A_union = (A_union + A_union')/2;
A_union_bin = double(full(A_union > 0));

if has_gt
    col_names = {'Seed','Silhouette','CH','DB','MQ_emb','MQ_graph','FinalLoss',...
                 'MoJoFM','TurboMQ','Cohesion','Coupling'};
    n_cols = 11;
else
    col_names = {'Seed','Silhouette','CH','DB','MQ_emb','MQ_graph','FinalLoss'};
    n_cols = 7;
end
results = zeros(n_seeds, n_cols);
all_labels = cell(n_seeds,1);
all_embeddings = cell(n_seeds,1);
all_loss_history = cell(n_seeds,1);
timestamp = datestr(now,'yyyy-mm-dd_HH-MM-SS');

B1=0.9; B2=0.999; EA=1e-8;
hid = max(128, cfg.dr*2);

for si = 1:n_seeds
    seed = seed_list(si);
    rng(seed);
    if cfg.verbose, fprintf('\n--- Seed %d (%d/%d) ---\n', seed, si, n_seeds); end

    if strcmp(cfg.k_selection_method, 'silhouette')
        X_quick = Xv_list{v_dom_init};
        X_quick = X_quick ./ (sqrt(sum(X_quick.^2,2)) + eps);
        best_k = cfg.k_range(1);
        best_sil = -inf;
        for ktest = cfg.k_range
            if ktest >= N, continue; end
            lbl = kmeans(X_quick, ktest, 'Replicates', 5, 'MaxIter', 300, 'EmptyAction','singleton');
            sil_vals = silhouette(X_quick, lbl);
            sil_mean = mean(sil_vals);
            if sil_mean > best_sil
                best_sil = sil_mean;
                best_k = ktest;
            end
        end
        C_opt = best_k;
        if cfg.verbose, fprintf('  Selected K_opt = %d (silhouette=%.4f)\n', C_opt, best_sil); end
    else
        C_opt = cfg.k_fixed;
    end
    cfg_current = cfg;
    cfg_current.C = C_opt;

    W1 = glorot(D, hid);      b1 = zeros(1, hid);
    W2 = glorot(hid, cfg.dr); b2 = zeros(1, cfg.dr);
    Wd2 = glorot(cfg.dr, hid); bd2 = zeros(1, hid);
    Wd1 = glorot(hid, D);      bd1 = zeros(1, D);
    Wp = cell(1,V); bp = cell(1,V);
    for v=1:V
        Wp{v} = glorot(cfg.dr, cfg.dr); bp{v} = zeros(1, cfg.dr);
    end

    [mW1,vW1] = deal(zeros(size(W1))); [mb1,vb1] = deal(zeros(size(b1)));
    [mW2,vW2] = deal(zeros(size(W2))); [mb2,vb2] = deal(zeros(size(b2)));
    [mWd2,vWd2] = deal(zeros(size(Wd2))); [mbd2,vbd2] = deal(zeros(size(bd2)));
    [mWd1,vWd1] = deal(zeros(size(Wd1))); [mbd1,vbd1] = deal(zeros(size(bd1)));
    mWp = cell(1,V); vWp = cell(1,V); mbp = cell(1,V); vbp = cell(1,V);
    for v=1:V
        mWp{v}=zeros(size(Wp{v})); vWp{v}=zeros(size(Wp{v}));
        mbp{v}=zeros(1,cfg.dr); vbp{v}=zeros(1,cfg.dr);
    end
    adam_t = 0;
    v_dom = v_dom_init;
    L_last = NaN;

    opts0 = statset('MaxIter',cfg.max_km_iter,'UseParallel',false);
    y_hat = kmeans(Xv_list{v_dom}, C_opt, 'Replicates',3, 'Options',opts0, 'EmptyAction','singleton');

    loss_history = zeros(cfg.epochs, 4);

    for ep = 1:cfg.epochs
        adam_t = adam_t+1;

        Z = cell(1,V); H1 = cell(1,V); Hd = cell(1,V); Xr = cell(1,V);
        for v = 1:V
            h1 = relu(Xv_list{v}*W1 + repmat(b1,N,1));
            zv = h1*W2 + repmat(b2,N,1);
            hd = relu(zv*Wd2 + repmat(bd2,N,1));
            xrec = hd*Wd1 + repmat(bd1,N,1);
            H1{v}=h1; Z{v}=zv; Hd{v}=hd; Xr{v}=xrec;
        end

        L_rec = 0; dXr = cell(1,V);
        for v = 1:V
            Xv = Xv_list{v};
            nX = sqrt(sum(Xv.^2,2)) + eps;
            nR = sqrt(sum(Xr{v}.^2,2)) + eps;
            csi = sum(Xv.*Xr{v},2) ./ (nX.*nR);
            L_rec = L_rec + mean(1-csi);
            dXr{v} = -(Xv./(nX.*nR) - Xr{v}.*csi./(nR.^2))/N;
        end
        L_rec = L_rec / V;

        if mod(ep, cfg.t_recalc) == 0
            v_dom = find_dominant_view(X, Z, V, N);
        end

        L_anf = 0; dZ_anf = cell(1,V);
        if N <= 3000
            XXt = X*X';
            for v = 1:V
                diff = XXt - Z{v}*Z{v}';
                L_anf = L_anf + sum(diff(:).^2)/N^2;
                dZ_anf{v} = (-4/N^2)*(diff*Z{v});
            end
        else
            n_sub = min(1000,N); idx_s = randperm(N,n_sub);
            XXts = X(idx_s,:)*X(idx_s,:)';
            for v = 1:V
                Zs = Z{v}(idx_s,:);
                diff = XXts - Zs*Zs';
                L_anf = L_anf + sum(diff(:).^2)/n_sub^2;
                gs = (-4/n_sub^2)*(diff*Zs);
                dZ_anf{v} = zeros(N,cfg.dr);
                dZ_anf{v}(idx_s,:) = gs;
            end
        end
        L_anf = L_anf / V;

        L_adv = 0;
        dZ_adv = cell(1,V); dWp_g = cell(1,V); dbp_g = cell(1,V);
        for v=1:V
            dZ_adv{v}=zeros(N,cfg.dr);
            dWp_g{v}=zeros(cfg.dr,cfg.dr);
            dbp_g{v}=zeros(1,cfg.dr);
        end

        if V > 1
            Zt = cell(1,V);
            for v=1:V
                Zt{v} = l2norm(Z{v}*Wp{v} + repmat(bp{v},N,1));
            end
            coeff = 1/(2*N*(V-1));
            for v=1:V
                if v == v_dom, continue; end
                S1 = (Zt{v}*Zt{v_dom}')/cfg.tau;
                S1 = S1 - max(S1,[],2);
                eS1 = exp(S1); d1 = sum(eS1,2)+eps;
                L_adv = L_adv + coeff*(-mean(log(diag(eS1)./(d1)+eps)));
                P1 = eS1./d1 - eye(N)/N;
                gv1 = (P1*Zt{v_dom})/(cfg.tau*N);
                gd1 = (P1'*Zt{v})/(cfg.tau*N);
                S2 = (Zt{v_dom}*Zt{v}')/cfg.tau;
                S2 = S2 - max(S2,[],2);
                eS2 = exp(S2); d2 = sum(eS2,2)+eps;
                L_adv = L_adv + coeff*(-mean(log(diag(eS2)./(d2)+eps)));
                P2 = eS2./d2 - eye(N)/N;
                gd2 = (P2*Zt{v})/(cfg.tau*N);
                gv2 = (P2'*Zt{v_dom})/(cfg.tau*N);
                gZp_v = coeff * bprop_l2(Zt{v}, gv1+gv2);
                gZp_dom = coeff * bprop_l2(Zt{v_dom}, gd1+gd2);
                dWp_g{v} = dWp_g{v} + Z{v}'*gZp_v;
                dbp_g{v} = dbp_g{v} + sum(gZp_v,1);
                dZ_adv{v} = dZ_adv{v} + gZp_v*Wp{v}';
                dWp_g{v_dom} = dWp_g{v_dom} + Z{v_dom}'*gZp_dom/(V-1);
                dbp_g{v_dom} = dbp_g{v_dom} + sum(gZp_dom,1)/(V-1);
                dZ_adv{v_dom} = dZ_adv{v_dom} + gZp_dom*Wp{v_dom}'/(V-1);
            end
        end

        L_cal = L_adv + L_anf;

        if mod(ep, cfg.t_recalc) == 0
            opts_km = statset('MaxIter',cfg.max_km_iter,'UseParallel',false);
            y_hat = kmeans(Z{v_dom}, C_opt, 'Replicates',3, 'Options',opts_km, 'EmptyAction','singleton');
        end

        Zcat = cat(2, Z{:});
        all_Zv = [Z, {Zcat}];
        n_vt = numel(all_Zv);
        Q_all = cell(1,n_vt); P_all = cell(1,n_vt);
        for zi = 1:n_vt
            Zzi = all_Zv{zi};
            dz = size(Zzi,2);
            sig = zeros(C_opt, dz);
            for j = 1:C_opt
                ij = (y_hat == j);
                if any(ij), sig(j,:) = mean(Zzi(ij,:),1); end
            end
            Qz = zeros(N, C_opt);
            for j = 1:C_opt
                df = Zzi - sig(j,:);
                Qz(:,j) = 1./(1+sum(df.^2,2));
            end
            Qz = Qz ./ (sum(Qz,2)+eps);
            Qs = sum(Qz,1)+eps;
            Pz = (Qz.^2) ./ Qs;
            Pz = Pz ./ (sum(Pz,2)+eps);
            Q_all{zi} = Qz; P_all{zi} = Pz;
        end

        L_clu = 0;
        for zi = 1:n_vt
            kl = sum(P_all{zi}.*log(P_all{zi}./(Q_all{zi}+eps)+eps), 'all') / N;
            if zi == n_vt, L_clu = L_clu + kl;
            else, L_clu = L_clu + kl/V; end
        end

        dZ_clu = cell(1,V);
        for v=1:V, dZ_clu{v}=zeros(N,cfg.dr); end
        for v=1:V
            Zvi = Z{v}; Qvi = Q_all{v}; Pvi = P_all{v};
            sigv = zeros(C_opt, cfg.dr);
            for j=1:C_opt
                ij = (y_hat == j);
                if any(ij), sigv(j,:) = mean(Zvi(ij,:),1); end
            end
            Qun = zeros(N, C_opt);
            for j=1:C_opt
                df = Zvi - sigv(j,:);
                Qun(:,j) = 1./(1+sum(df.^2,2));
            end
            Qsr = sum(Qun,2)+eps;
            dKL = -Pvi./(Qvi+eps) / (V*N);
            dZv = zeros(N,cfg.dr);
            for j=1:C_opt
                df = Zvi - sigv(j,:);
                qij = Qun(:,j)./Qsr;
                cj = dKL(:,j) .* (-2*qij.*(1-qij));
                dZv = dZv + cj .* df;
            end
            dZ_clu{v} = dZv;
        end

        L_total = L_rec + L_cal + L_clu;
        L_last = L_total;
        loss_history(ep,:) = [L_total, L_rec, L_cal, L_clu];
        if isnan(L_total)
            warning('NaN loss at seed=%d, epoch=%d', seed, ep); break;
        end
        if cfg.verbose && mod(ep, max(1,floor(cfg.epochs/5)))==0
            fprintf(' [seed %2d] ep %3d/%d | L=%.4f (rec=%.4f cal=%.4f clu=%.4f) v*=%d\n',...
                    seed, ep, cfg.epochs, L_total, L_rec, L_cal, L_clu, v_dom);
        end

        gW1 = zeros(size(W1)); gb1 = zeros(size(b1));
        gW2 = zeros(size(W2)); gb2 = zeros(size(b2));
        gWd2 = zeros(size(Wd2)); gbd2 = zeros(size(bd2));
        gWd1 = zeros(size(Wd1)); gbd1 = zeros(size(bd1));
        for v=1:V
            gWd1 = gWd1 + Hd{v}'*dXr{v};
            gbd1 = gbd1 + sum(dXr{v},1);
            dHd = dXr{v}*Wd1'; dHd(Hd{v}<=0)=0;
            gWd2 = gWd2 + Z{v}'*dHd;
            gbd2 = gbd2 + sum(dHd,1);
            dZdec = dHd*Wd2';
            dZv = dZdec + dZ_adv{v} + dZ_anf{v} + dZ_clu{v};
            gW2 = gW2 + H1{v}'*dZv;
            gb2 = gb2 + sum(dZv,1);
            dH1 = dZv*W2'; dH1(H1{v}<=0)=0;
            gW1 = gW1 + Xv_list{v}'*dH1;
            gb1 = gb1 + sum(dH1,1);
        end

        [W1,mW1,vW1] = adam(W1,gW1,mW1,vW1,cfg.lr,cfg.weight_decay,B1,B2,EA,adam_t);
        [b1,mb1,vb1] = adam(b1,gb1,mb1,vb1,cfg.lr,0,B1,B2,EA,adam_t);
        [W2,mW2,vW2] = adam(W2,gW2,mW2,vW2,cfg.lr,cfg.weight_decay,B1,B2,EA,adam_t);
        [b2,mb2,vb2] = adam(b2,gb2,mb2,vb2,cfg.lr,0,B1,B2,EA,adam_t);
        [Wd2,mWd2,vWd2] = adam(Wd2,gWd2,mWd2,vWd2,cfg.lr,cfg.weight_decay,B1,B2,EA,adam_t);
        [bd2,mbd2,vbd2] = adam(bd2,gbd2,mbd2,vbd2,cfg.lr,0,B1,B2,EA,adam_t);
        [Wd1,mWd1,vWd1] = adam(Wd1,gWd1,mWd1,vWd1,cfg.lr,cfg.weight_decay,B1,B2,EA,adam_t);
        [bd1,mbd1,vbd1] = adam(bd1,gbd1,mbd1,vbd1,cfg.lr,0,B1,B2,EA,adam_t);
        for v=1:V
            [Wp{v},mWp{v},vWp{v}] = adam(Wp{v},dWp_g{v},mWp{v},vWp{v},cfg.lr,cfg.weight_decay,B1,B2,EA,adam_t);
            [bp{v},mbp{v},vbp{v}] = adam(bp{v},dbp_g{v},mbp{v},vbp{v},cfg.lr,0,B1,B2,EA,adam_t);
        end
    end

    Zf = cell(1,V);
    for v=1:V
        h1 = relu(Xv_list{v}*W1 + repmat(b1,N,1));
        Zf{v} = h1*W2 + repmat(b2,N,1);
    end
    Zfinal = cat(2, Zf{:});
    all_embeddings{si} = Zfinal;

    opts_km = statset('MaxIter',cfg.max_km_iter,'UseParallel',false);
    labels = kmeans(Zfinal, C_opt, 'Replicates', cfg.kmeans_reps,...
                    'Options', opts_km, 'EmptyAction','singleton');
    all_labels{si} = labels;
    all_loss_history{si} = loss_history;

    if N > 5000
        idx_s = randperm(N,5000);
        sil = mean(silhouette(Zfinal(idx_s,:), labels(idx_s)));
    else
        sil = mean(silhouette(Zfinal, labels));
    end
    try
        ev = evalclusters(Zfinal, labels, 'CalinskiHarabasz');
        chv = ev.CriterionValues;
    catch; chv = NaN; end
    try; dbv = compute_DB(Zfinal, labels); catch; dbv = NaN; end
    mq_emb = compute_MQ_emb(Zfinal, labels);
    mq_graph = compute_MQ_graph(A_union, labels);

    if has_gt
        mojofm = compute_MoJoFM(gt_labels, labels);
        turbomq = compute_TurboMQ(A_union_bin, labels);
        [cohesion, coupling] = compute_cohesion_coupling(A_union_bin, labels);
    else
        mojofm = NaN; turbomq = NaN; cohesion = NaN; coupling = NaN;
    end

    if has_gt
        results(si,:) = [seed, sil, chv, dbv, mq_emb, mq_graph, L_last, mojofm, turbomq, cohesion, coupling];
        fprintf(' Seed %2d -> Sil=%.4f | CH=%.2f | DB=%.4f | MQ_emb=%.4f | MQ_graph=%.4f | MoJoFM=%.4f | TurboMQ=%.4f | Coh=%.4f | Coupl=%.4f\n',...
                seed, sil, chv, dbv, mq_emb, mq_graph, mojofm, turbomq, cohesion, coupling);
    else
        results(si,:) = [seed, sil, chv, dbv, mq_emb, mq_graph, L_last];
        fprintf(' Seed %2d -> Sil=%.4f | CH=%.2f | DB=%.4f | MQ_emb=%.4f | MQ_graph=%.4f\n',...
                seed, sil, chv, dbv, mq_emb, mq_graph);
    end

    seed_file = fullfile(cfg.outdir, sprintf('bmgc_seed%d.mat', seed));
    if has_gt
        save(seed_file, 'Zfinal', 'labels', 'loss_history', 'cfg', 'sil', 'chv', 'dbv', 'mq_emb', 'mq_graph', 'L_last', ...
             'mojofm', 'turbomq', 'cohesion', 'coupling', 'C_opt');
    else
        save(seed_file, 'Zfinal', 'labels', 'loss_history', 'cfg', 'sil', 'chv', 'dbv', 'mq_emb', 'mq_graph', 'L_last', 'C_opt');
    end
end

results_table = array2table(results, 'VariableNames', col_names);

fprintf('\n=================================================================\n');
fprintf(' SUMMARY — Seeds %d to %d (%d runs)\n', cfg.seed_start, cfg.seed_end, n_seeds);
fprintf('=================================================================\n');
fprintf(' %-12s  %8s  %8s  %8s\n', 'Metric', 'Mean', 'Std', 'Best');
fprintf(' %s\n', repmat('-',1,46));
metric_info = {'Silhouette',2,'max'; 'CH',3,'max'; 'DB',4,'min';...
               'MQ_emb',5,'max'; 'MQ_graph',6,'max'};
if has_gt
    metric_info = [metric_info; {'MoJoFM',8,'max'}; {'TurboMQ',9,'max'}; {'Cohesion',10,'max'}; {'Coupling',11,'min'}];
end
for mi = 1:size(metric_info,1)
    col_idx = metric_info{mi,2};
    vals = results(:,col_idx);
    if strcmp(metric_info{mi,3}, 'max'), best = max(vals); else, best = min(vals); end
    fprintf(' %-12s  %8.4f  %8.4f  %8.4f\n', metric_info{mi,1}, mean(vals,'omitnan'), std(vals,'omitnan'), best);
end
[~, bsi] = max(results(:,2));
fprintf('\n Best seed by Silhouette: Seed %d\n', results(bsi,1));

if cfg.save_csv
    fn_csv = fullfile(cfg.outdir, sprintf('bmgc_results_%s.csv', timestamp));
    fn_mat = fullfile(cfg.outdir, sprintf('bmgc_results_%s.mat', timestamp));
    writetable(results_table, fn_csv);
    save(fn_mat, 'results_table', 'all_labels', 'all_embeddings', 'all_loss_history', 'cfg', '-v7.3');
    fprintf(' Saved: %s | %s\n', fn_csv, fn_mat);
end
fprintf('=================================================================\n');
end

function vd = find_dominant_view(X, Z_list, V, N)
if N <= 5000
    XXt = X*X'; frob = zeros(1,V);
    for v=1:V
        d = XXt - Z_list{v}*Z_list{v}'; frob(v) = sum(d(:).^2);
    end
else
    n_s = min(500,N); idx = randperm(N, n_s);
    XXt = X(idx,:)*X(idx,:)'; frob = zeros(1,V);
    for v=1:V
        Zs = Z_list{v}(idx,:); d = XXt - Zs*Zs'; frob(v) = sum(d(:).^2);
    end
end
[~, vd] = min(frob);
end

function W = glorot(fi, fo)
W = (2*rand(fi,fo)-1)*sqrt(6/(fi+fo));
end

function Y = relu(X), Y = max(0,X); end

function Xn = l2norm(X)
Xn = X ./ (sqrt(sum(X.^2,2)) + eps);
end

function dX = bprop_l2(Xn, dOut)
dX = dOut - Xn .* sum(dOut.*Xn,2);
end

function [p,m,v] = adam(p,g,m,v,lr,wd,b1,b2,ep,t)
g = g + wd*p;
m = b1*m + (1-b1)*g;
v = b2*v + (1-b2)*g.^2;
p = p - lr*(m/(1-b1^t)) ./ (sqrt(v/(1-b2^t)) + ep);
end

function db = compute_DB(Z, labels)
C = unique(labels); K = numel(C);
ct = zeros(K, size(Z,2)); S = zeros(K,1);
for i = 1:K
    ix = (labels == C(i));
    ct(i,:) = mean(Z(ix,:),1);
    if sum(ix) > 1
        S(i) = mean(sqrt(sum((Z(ix,:)-ct(i,:)).^2,2)));
    end
end
M = pdist2(ct,ct); M(M==0)=inf;
R = zeros(K,K);
for i=1:K, for j=1:K
    if i~=j, R(i,j) = (S(i)+S(j))/M(i,j); end
end; end
db = mean(max(R,[],2));
end

function MQ = compute_MQ_emb(Z, clusters)
K = numel(unique(clusters));
MQk = zeros(K,1);
for k=1:K
    idx = find(clusters == k);
    if numel(idx) > 1
        MQk(k) = 1 - mean(pdist(Z(idx,:),'cosine'));
    end
end
MQ = mean(MQk);
end

function MQ = compute_MQ_graph(A, labels)
A = full(A);
Cu = unique(labels); K = numel(Cu);
mu = zeros(K,1); ep = zeros(K,K);
for i=1:K
    ii = (labels == Cu(i)); ni = sum(ii);
    if ni >= 2
        mu(i) = (sum(sum(A(ii,ii)))/2) / (ni*(ni-1)/2);
    end
    for j=i+1:K
        ij = (labels == Cu(j)); nj = sum(ij);
        inter = sum(sum(A(ii,ij)));
        ep(i,j) = inter/(ni*nj); ep(j,i) = ep(i,j);
    end
end
MQk = zeros(K,1);
for i=1:K
    MQk(i) = mu(i) - 0.5*sum(ep(i,:));
end
MQ = mean(MQk);
end

function mfm = compute_MoJoFM(true_labels, pred_labels)
true_labels = true_labels(:);
pred_labels = pred_labels(:);
n = length(true_labels);

[~, ~, true_idx] = unique(true_labels);
[~, ~, pred_idx] = unique(pred_labels);
K_true = max(true_idx);
K_pred = max(pred_idx);

C = accumarray([true_idx, pred_idx], 1, [K_true, K_pred]);

cost = -C;
m = size(cost,1);
ncol = size(cost,2);
if m > ncol
    cost = [cost, zeros(m, m - ncol)];
elseif ncol > m
    cost = [cost; zeros(ncol - m, ncol)];
end

try
    assignment = matchpairs(cost, max(cost(:))*10);
    matched = zeros(K_true, 1);
    for i = 1:size(assignment,1)
        if assignment(i,1) <= K_true && assignment(i,2) <= K_pred
            matched(assignment(i,1)) = assignment(i,2);
        end
    end
    total_correct = 0;
    for i = 1:K_true
        if matched(i) > 0
            total_correct = total_correct + C(i, matched(i));
        end
    end
    mfm = total_correct / n;
catch
    [vals, idx] = sort(C(:), 'descend');
    matched = zeros(K_true, 1);
    used = false(1, K_pred);
    for k = 1:numel(vals)
        if vals(k) == 0, break; end
        [r, c] = ind2sub([K_true, K_pred], idx(k));
        if ~matched(r) && ~used(c)
            matched(r) = c;
            used(c) = true;
        end
    end
    total_correct = 0;
    for i = 1:K_true
        if matched(i) > 0
            total_correct = total_correct + C(i, matched(i));
        end
    end
    mfm = total_correct / n;
end
end

function tmq = compute_TurboMQ(A_bin, clusters)
A_bin = double(A_bin > 0);
uq = unique(clusters);
K = numel(uq);
mqk = zeros(K,1); w = zeros(K,1);
for k = 1:K
    idx = find(clusters == uq(k));
    other = find(clusters ~= uq(k));
    I = sum(A_bin(idx,idx), 'all');
    E = sum(A_bin(idx,other), 'all');
    if I+E > 0, mqk(k) = I/(I+E); end
    w(k) = length(idx);
end
tmq = sum(w .* mqk) / sum(w);
end

function [avg_cohesion, avg_coupling] = compute_cohesion_coupling(A_bin, clusters)
A_bin = double(A_bin > 0);
uq = unique(clusters);
K = numel(uq);
coh = zeros(K,1); coup = zeros(K,1);
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
avg_cohesion = mean(coh);
avg_coupling = mean(coup);
end