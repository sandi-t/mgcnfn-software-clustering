clc; clear;

dataset_name = 'javacc';
num_packages = 4;

data_dir = './javacc/';
feat_file = [data_dir 'node_features.txt'];
label_file = [data_dir 'label.txt'];

N_SEEDS = 50;

X = load(feat_file);
[N, D] = size(X);

X = normalize(X, 2);

has_gt = false;
gt_labels = [];
if isfile(label_file)
    gt_labels = load(label_file);
    gt_labels = gt_labels(:);
    if length(gt_labels) == N
        has_gt = true;
    else
        warning('Label count mismatch');
    end
end

if has_gt
    results = zeros(N_SEEDS, 8);
    metric_names = {'Seed','Silhouette','CH','DB','NMI','ARI','MoJoFM','k'};
else
    results = zeros(N_SEEDS, 5);
    metric_names = {'Seed','Silhouette','CH','DB','k'};
end

fprintf('=================================================================\n');
fprintf(' K-Means Clustering (fixed k=%d, %d seeds)\n', num_packages, N_SEEDS);
fprintf('=================================================================\n');

for s = 1:N_SEEDS
    rng(s, 'twister');
    
    [labels, ~] = kmeans(X, num_packages, ...
        'Replicates', 20, 'MaxIter', 500, 'Distance', 'cosine', ...
        'Start', 'plus', 'Display', 'off');
    
    sil = mean(silhouette(X, labels));
    try
        ev = evalclusters(X, labels, 'CalinskiHarabasz');
        ch = ev.CriterionValues;
    catch
        ch = NaN;
    end
    try
        ev = evalclusters(X, labels, 'DaviesBouldin');
        db = ev.CriterionValues;
    catch
        db = NaN;
    end
    
    if has_gt
        nmi = computeNMI(gt_labels, labels);
        ari = computeARI(gt_labels, labels);
        mojofm = compute_MoJoFM(gt_labels, labels);
        results(s,:) = [s, sil, ch, db, nmi, ari, mojofm, num_packages];
    else
        results(s,:) = [s, sil, ch, db, num_packages];
    end
    
    if mod(s,10)==0 || s==1
        if has_gt
            fprintf(' Seed %3d: Sil=%.4f | CH=%.2f | DB=%.4f | NMI=%.4f | ARI=%.4f | MoJoFM=%.4f\n', ...
                s, sil, ch, db, nmi, ari, mojofm);
        else
            fprintf(' Seed %3d: Sil=%.4f | CH=%.2f | DB=%.4f\n', s, sil, ch, db);
        end
    end
end

fprintf('\n=================================================================\n');
fprintf(' SUMMARY over %d seeds (k=%d)\n', N_SEEDS, num_packages);
fprintf('=================================================================\n');
fprintf(' %-12s  %8s  %8s  %8s\n', 'Metric', 'Mean', 'Std', 'Best');
fprintf(' %s\n', repmat('-',1,46));

if has_gt
    metric_info = {'Silhouette',2,'max'; 'CH',3,'max'; 'DB',4,'min'; ...
                   'NMI',5,'max'; 'ARI',6,'max'; 'MoJoFM',7,'max'};
else
    metric_info = {'Silhouette',2,'max'; 'CH',3,'max'; 'DB',4,'min'};
end

for i = 1:size(metric_info,1)
    name = metric_info{i,1};
    col  = metric_info{i,2};
    vals = results(:,col);
    if strcmp(metric_info{i,3}, 'max')
        best = max(vals);
    else
        best = min(vals);
    end
    fprintf(' %-12s  %8.4f  %8.4f  %8.4f\n', name, mean(vals), std(vals), best);
end
fprintf('=================================================================\n');

timestamp = datestr(now,'yyyymmdd_HHMMSS');
outfile = sprintf('kmeans_results_%s_%s.csv', dataset_name, timestamp);
writetable(array2table(results, 'VariableNames', metric_names), outfile);
fprintf('Results saved to %s\n', outfile);

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
    for k = 1:numel(vals)
        if vals(k) == 0, break; end
        [r,c] = ind2sub([Kt, Kp], idx(k));
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