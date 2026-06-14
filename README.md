# MGCNFN – Multi‑Graph Convolutional Normalizing Flows for Software Module Clustering

This repository contains the MATLAB source code for MGCNFN and the baseline methods used in our paper. The replication package includes all scripts needed to reproduce the tables and figures, as well as raw per‑seed results.

## Data Availability

The datasets, feature matrices, and relation‑specific adjacency matrices (association, aggregation, composition, dependency) are available on Figshare:  
(https://doi.org/10.6084/m9.figshare.30932816)

Place the downloaded data in the corresponding folders (e.g., `./jhotdraw/`, `./jedit/`) before running the scripts.

## KNN Graph Construction Code

The KNN graph is built using cosine similarity. The implementation is inside the main MGCNFN script (`mgcnfn.m`) in a local function named `build_knn`. The function computes the pairwise cosine similarity between preprocessed feature vectors, then for each node selects its `K` most similar neighbours (excluding the node itself) and adds undirected edges. The adjacency matrix is symmetrised by `A | A'`.

The number of neighbours is set as a configuration parameter:

```matlab
cfg.K_knn = 4;
The function is called after PCA dimensionality reduction and before constructing the normalised adjacency matrices.

Hyperparameter Configuration
All hyperparameters are defined as fields of a structure named cfg at the beginning of each main script. No separate configuration files are used, so all settings are explicit and can be directly inspected.

MGCNFN (mgcnfn.m)
The configuration block appears near the top of the file. Key parameters include:
cfg.encoder_dims = [512, 256];   % hidden layers of the autoencoder
cfg.latent_dim = 64;             % bottleneck dimension, also equals gcn_hidden
cfg.gcn_hidden = 64;             % output dimension of each GCN branch
cfg.gcn_layers = 2;              % number of GCN layers per branch
cfg.att_h = 64;                  % hidden dimension for multi‑graph attention
cfg.alpha = 1.0;                 % weight for reconstruction loss
cfg.beta = 1e-4;                 % weight for consistency loss
cfg.flow.numBlocks = 4;          % number of affine coupling blocks
cfg.flow.hidden = 64;            % hidden units in scale/translation networks
cfg.flow.mp_r = 32;              % hidden dimension for message passing
cfg.flow.lambda = 0.01;          % weight for the flow loss
cfg.maxEpochs = 300;             % number of training epochs
cfg.lr = 1e-3;                   % initial learning rate (halved every 50 epochs)
cfg.grad_clip = 1.0;             % gradient clipping threshold


GC‑Flows (gcflow_software_clustering.m)
Key hyperparameters:
cfg.gcn_hidden = 128;            % first GCN layer dimension
cfg.latent_dim = 64;             % output dimension before flow
cfg.flow.numBlocks = 4;          % number of coupling blocks
cfg.flow.hidden = 64;            % hidden size in s‑net and t‑net
cfg.gmm.K = 7;                   % number of GMM components
cfg.maxEpochs = 400;             % maximum epochs (early stopping with patience 30)
cfg.lr = 1e-3;
cfg.grad_clip = 50;


BMGC (bmgc_software_clustering.m)
Key hyperparameters:
cfg.C = 5;                       % number of clusters (not used for final evaluation)
cfg.epochs = 300;
cfg.dr = 64;                     % embedding dimension
cfg.K = 3;                       % propagation steps
cfg.alpha = 0.2;                 % teleport parameter
cfg.t_recalc = 10;               % interval for updating dominant view
cfg.lr = 1e-2;
cfg.weight_decay = 1e-4;


MGCCN (mgccn_software_clustering.m)
Key hyperparameters:
cfg.latent_dim = 64;             % also equals gcn_hidden
cfg.encoder_dims = [512, 256];
cfg.gcn_layers = 2;
cfg.att_h = 64;
cfg.alpha = 1.0;
cfg.beta = 1e-4;
cfg.epsilon = 0.5;               % residual injection coefficient for KNN branch
cfg.K = 8;                       % KNN graph parameter (different from MGCNFN)
cfg.maxEpochs = 300;
cfg.lr = 1e-3;
List of 50 Random Seeds

All experiments use the same 50 random seeds, from 1 to 50. The seed range is defined in the configuration section of each script:
cfg.seed_start = 1;
cfg.seed_end = 50;
The main loop then iterates over these values:

for seed = cfg.seed_start : cfg.seed_end
    rng(seed, 'twister');
    % ... run experiment for this seed
end

This applies to MGCNFN, GC‑Flows, BMGC, and MGCCN. For K‑means, which is called from within the evaluation scripts, the same set of seeds is used for reproducibility.

## Raw Per‑Seed Results

The raw results (50 values per seed) are organised in the following folder structure:
raw_results/
├── BMGC/

│ ├── BMGC_JavaCC.csv
│ ├── BMGC_Dom4J.csv
│ ├── BMGC_JUnit.csv
│ ├── BMGC_JHotDraw.csv
│ └── BMGC_JEdit.csv
├── GC-Flows/
│ ├── GC-Flows_JavaCC.csv
│ └── ...
├── K-means/
│ ├── K-means_JavaCC.csv
│ └── ...
├── MGCCN/
│ ├── MGCCN_JavaCC.csv
│ └── ...
└── MGCNFN/
├── MGCNFN_JavaCC.csv
└── ...


Each CSV file contains the following columns for 50 seeds (one row per seed):

| Column | Description |
|--------|-------------|
| `seed` | Random seed number (1 to 50) |
| `Silhouette` | Silhouette coefficient |
| `CH` | Calinski‑Harabasz Index |
| `DB` | Davies‑Bouldin Index |
| `MQ` | Modularization Quality |
| `MoJoFM` | MoJoFM |
| `TurboMQ` | Turbo Modularization Quality |
| `Cohesion` | Internal cluster cohesion |
| `Coupling` | Inter‑cluster coupling |

For example, the first few rows of `BMGC_Dom4J.csv` look like:

```csv
seed,Silhouette,CH,DB,MQ,MoJoFM,TurboMQ,Cohesion,Coupling
1,0.25347,28.6618,1.81858,0.12718,0.33728,0.25541,0.16968,0.74604
2,0.24866,32.4170,1.79405,0.23601,0.35503,0.28906,0.19673,0.70812
3,0.24382,32.5072,1.74486,0.28670,0.37870,0.27940,0.19393,0.72316
...


License
The source code is released under the CC‑BY 4.0 license. You may use, share, and adapt it as long as proper attribution is given.

Contact
For questions or issues, please open an issue on this GitHub repository or contact the corresponding author (email in the paper).
