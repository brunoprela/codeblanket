#!/usr/bin/env python3
"""
Generate all Module 9 quiz files with comprehensive discussion questions
"""

import os

# Create directory if needed
os.makedirs("lib/content/quizzes/ml-unsupervised-learning", exist_ok=True)

quizzes = {
    "unsupervised-learning-overview": [
        {
            "question": "What are the fundamental differences between supervised and unsupervised learning, and in what scenarios would you choose unsupervised learning over supervised learning?",
            "hint": "Consider the availability of labels, the goal of analysis, and typical applications.",
            "sampleAnswer": "Supervised learning requires labeled training data where both inputs and desired outputs are known, while unsupervised learning works with unlabeled data to discover hidden patterns. Unsupervised learning is chosen when: (1) Labels are unavailable, expensive, or impractical to obtain, (2) The goal is exploratory data analysis or pattern discovery rather than prediction, (3) You want to reduce dimensionality before supervised learning, (4) You need to detect anomalies or outliers, (5) You want to segment customers or group similar items without predefined categories. Supervised learning is preferred when you have clear target variables and sufficient labeled data for training.",
            "keyPoints": [
                "Supervised learning requires labeled data; unsupervised does not",
                "Unsupervised is used for pattern discovery and exploration",
                "Common when labels are expensive or unavailable",
                "Often used for preprocessing (dimensionality reduction)",
                "Suitable for clustering, anomaly detection, and segmentation"
            ]
        },
        {
            "question": "Explain the curse of dimensionality and its implications for unsupervised learning algorithms. How do dimensionality reduction techniques help mitigate this problem?",
            "hint": "Consider distance metrics, data sparsity, and computational complexity.",
            "sampleAnswer": "The curse of dimensionality refers to various phenomena that arise when analyzing data in high-dimensional spaces. As dimensions increase: (1) Distances become less meaningful - all points appear equally far apart, (2) Data becomes increasingly sparse - exponentially more data needed to maintain density, (3) Computational complexity explodes - algorithms scale poorly, (4) Visualization becomes impossible beyond 3D. For unsupervised learning, this affects clustering (distances lose meaning), anomaly detection (everything looks like an outlier), and pattern discovery. Dimensionality reduction techniques like PCA, t-SNE, and UMAP help by: (1) Removing redundant/correlated features, (2) Projecting to lower dimensions while preserving structure, (3) Reducing computational cost, (4) Enabling visualization, (5) Removing noise. This makes algorithms more effective and interpretable.",
            "keyPoints": [
                "High dimensions make distances less meaningful",
                "Data becomes increasingly sparse",
                "Computational complexity increases exponentially",
                "Affects clustering, anomaly detection, visualization",
                "Dimensionality reduction preserves structure while reducing dimensions"
            ]
        },
        {
            "question": "Compare and contrast the three main categories of unsupervised learning: clustering, dimensionality reduction, and anomaly detection. Provide real-world examples for each.",
            "hint": "Focus on objectives, outputs, and typical use cases.",
            "sampleAnswer": "The three main categories serve different purposes: CLUSTERING groups similar items together, producing discrete group assignments. Output is cluster labels. Used for: customer segmentation (grouping customers by behavior), document organization (grouping similar articles), image segmentation (grouping pixels). DIMENSIONALITY REDUCTION transforms high-D data to lower-D while preserving structure. Output is reduced feature set or projection. Used for: visualization (plotting high-D data in 2D/3D), preprocessing (reducing features before supervised learning), compression (storing images efficiently). ANOMALY DETECTION identifies unusual patterns. Output is anomaly scores or binary flags. Used for: fraud detection (identifying suspicious transactions), intrusion detection (finding network attacks), quality control (detecting defective products). They can be combined: reduce dimensions with PCA, then cluster with K-Means, then detect anomalies in clusters.",
            "keyPoints": [
                "Clustering: groups similar items, outputs cluster labels",
                "Dimensionality reduction: reduces features, outputs projections",
                "Anomaly detection: finds outliers, outputs anomaly scores",
                "Each serves distinct purposes with different applications",
                "Can be combined for comprehensive analysis"
            ]
        }
    ],
    "k-means-clustering": [
        {
            "question": "Explain the K-Means algorithm step-by-step. What are the key assumptions K-Means makes about cluster structure, and how do these assumptions limit its applicability?",
            "hint": "Cover initialization, assignment, update steps, and assumptions about cluster shapes.",
            "sampleAnswer": "K-Means algorithm: (1) Initialize K centroids randomly, (2) Assign each point to nearest centroid (Euclidean distance), (3) Update centroids to mean of assigned points, (4) Repeat steps 2-3 until convergence (centroids stop moving). Key assumptions: (1) SPHERICAL CLUSTERS: assumes clusters are roughly circular/spherical - fails on non-convex shapes like crescents, (2) SIMILAR SIZE: tends to create equal-sized clusters even if true clusters vary, (3) SIMILAR DENSITY: struggles with varying density clusters, (4) EUCLIDEAN DISTANCE: assumes isotropic variance - fails if features have different scales or correlations, (5) NUMBER K KNOWN: requires specifying K upfront. Limitations: Cannot find moon-shaped clusters, elongated clusters, or clusters with holes. Use DBSCAN for arbitrary shapes, hierarchical clustering if K unknown, GMM for probabilistic assignments.",
            "keyPoints": [
                "Algorithm: initialize centroids, assign points, update centroids, repeat",
                "Assumes spherical, similar-sized clusters",
                "Uses Euclidean distance (sensitive to scaling)",
                "Requires K specified upfront",
                "Fails on non-convex shapes and varying densities"
            ]
        },
        {
            "question": "What is the elbow method for choosing K in K-Means? Why might it sometimes fail, and what alternative methods can be used to determine the optimal number of clusters?",
            "hint": "Explain WCSS, the elbow visualization, and discuss silhouette scores and other metrics.",
            "sampleAnswer": "Elbow method plots Within-Cluster Sum of Squares (WCSS) vs K. WCSS always decreases as K increases, but at diminishing rates. The 'elbow' - where improvement slows dramatically - suggests optimal K. LIMITATIONS: (1) Elbow may be ambiguous or not exist, (2) Subjective interpretation, (3) Doesn't account for cluster validity. ALTERNATIVES: (1) SILHOUETTE SCORE: measures how similar points are to their own cluster vs other clusters (-1 to 1, higher better). Clear maximum indicates optimal K. (2) DAVIES-BOULDIN INDEX: ratio of within-cluster to between-cluster distances (lower better). (3) GAP STATISTIC: compares WCSS to expected WCSS under null distribution. (4) CROSS-VALIDATION: if downstream task exists, choose K based on task performance. (5) DOMAIN KNOWLEDGE: sometimes K is known from business context. Best practice: try multiple methods and validate with domain experts.",
            "keyPoints": [
                "Elbow method: plot WCSS vs K, look for bend",
                "Can be ambiguous or non-existent",
                "Silhouette score measures cluster cohesion and separation",
                "Multiple methods exist (Gap statistic, Davies-Bouldin)",
                "Validate with domain knowledge and downstream performance"
            ]
        },
        {
            "question": "Explain the K-Means++ initialization algorithm. Why is it superior to random initialization, and how does it work mathematically?",
            "hint": "Cover the probability-based selection process and its advantages.",
            "sampleAnswer": "K-Means++ is a smart initialization that improves convergence and final results. ALGORITHM: (1) Choose first centroid uniformly at random from data points, (2) For each remaining centroid: calculate distance from each point to nearest chosen centroid, choose next centroid with probability proportional to distance squared: P(x) = D(x)²/Σ D(x)², (3) Repeat until K centroids chosen. ADVANTAGES: (1) SPREADS CENTROIDS: probabilistic selection ensures centroids start far apart, (2) FASTER CONVERGENCE: fewer iterations needed, (3) BETTER RESULTS: reduces chance of poor local minima, (4) THEORETICAL GUARANTEE: solution is O(log K) competitive with optimal. INTUITION: Points far from existing centroids have higher probability of being chosen, leading to better initial coverage. This is now the default in sklearn. Random initialization can lead to poor results that K-Means++ avoids by starting with well-distributed centroids.",
            "keyPoints": [
                "Chooses initial centroids probabilistically, not randomly",
                "Probability proportional to distance squared from nearest centroid",
                "Ensures centroids start far apart",
                "Faster convergence and better results",
                "Now default in most implementations"
            ]
        }
    ],
    "hierarchical-clustering": [
        {
            "question": "Compare and contrast the four main linkage methods (single, complete, average, Ward) in hierarchical clustering. When would you choose each method, and what are their respective advantages and disadvantages?",
            "hint": "Consider cluster shape preferences, sensitivity to outliers, and typical use cases.",
            "sampleAnswer": "SINGLE LINKAGE (minimum distance): merges clusters with closest pair of points. Pros: finds non-elliptical shapes, can handle chains. Cons: sensitive to noise/outliers, prone to 'chaining' effect (long, snake-like clusters). Use for: non-convex shapes when data is clean. COMPLETE LINKAGE (maximum distance): merges clusters with farthest pair closest. Pros: creates compact, spherical clusters; less sensitive to outliers. Cons: can break large clusters; biased toward equal size. Use for: spherical clusters, noisy data. AVERAGE LINKAGE: merges based on average pairwise distance. Pros: balanced approach; more robust than single; less biased than complete. Cons: computationally expensive. Use for: general-purpose clustering when you want balance. WARD: minimizes within-cluster variance. Pros: similar objective to K-Means; creates balanced clusters; most commonly used. Cons: only works with Euclidean distance; biased toward equal-sized clusters. Use for: general-purpose clustering, default choice. Ward is most popular; single for special shapes; complete for noise robustness.",
            "keyPoints": [
                "Single: minimum distance, finds chains, sensitive to noise",
                "Complete: maximum distance, compact clusters, breaks large clusters",
                "Average: balanced compromise, robust, more expensive",
                "Ward: minimizes variance, most popular, similar to K-Means",
                "Choice depends on data characteristics and cluster shape expectations"
            ]
        },
        {
            "question": "How do you interpret a dendrogram and use it to choose the number of clusters? What information does the height of merges provide?",
            "hint": "Explain how to 'cut' the dendrogram and what vertical distances mean.",
            "sampleAnswer": "A dendrogram is a tree showing hierarchical relationships. READING: X-axis shows data points/clusters; Y-axis shows distance/dissimilarity at which clusters merge; horizontal lines represent clusters; vertical lines show merges; longer vertical lines = more dissimilar clusters being merged. CHOOSING K: Look for large vertical gaps - these indicate natural separations. Cut horizontally through the dendrogram: number of vertical lines you cross = number of clusters. HEIGHT INTERPRETATION: Height represents distance between clusters being merged. Large jump in height suggests merging very dissimilar clusters (don't merge). PRACTICAL APPROACH: (1) Look for longest vertical lines without horizontal crossings (biggest gaps), (2) Cut below these gaps, (3) Count resulting clusters. Can also use inconsistency method: measures how inconsistent a merge is compared to merges at adjacent levels. High inconsistency = good place to cut. The dendrogram visualizes clustering at ALL scales simultaneously, unlike K-Means which commits to one K.",
            "keyPoints": [
                "Dendrogram shows hierarchical clustering at all scales",
                "Y-axis height represents distance between merging clusters",
                "Large vertical gaps suggest natural cluster boundaries",
                "Cut horizontally to obtain K clusters",
                "Provides visual interpretation K-Means cannot offer"
            ]
        },
        {
            "question": "Explain the time and space complexity of hierarchical clustering. Why doesn't it scale well to large datasets, and what strategies can be used to apply it to larger data?",
            "hint": "Cover distance matrix computation, algorithm complexity, and sampling strategies.",
            "sampleAnswer": "COMPLEXITY: Time: O(n³) naive, O(n² log n) with optimizations. Space: O(n²) to store distance matrix. SCALABILITY ISSUES: (1) Must compute and store full distance matrix (n² space), (2) Must consider all pairs at each merge (expensive), (3) Cannot parallelize easily, (4) Impractical for n > 10,000 samples. STRATEGIES FOR LARGE DATA: (1) SAMPLING: cluster representative sample, then assign remaining points to nearest cluster (loses some structure), (2) PCA PREPROCESSING: reduce dimensions first (faster distances), (3) MINI-BATCH: divide data into batches, cluster each, then cluster the cluster centers (hierarchical of hierarchical), (4) USE ALTERNATIVE: switch to K-Means (O(nKt)) or DBSCAN for large datasets, (5) APPROXIMATE METHODS: use approximate nearest neighbors. WHEN TO USE HIERARCHICAL: Best for n < 10,000, when dendrogram is valuable, when K is unknown, for exploratory analysis. For large datasets, use K-Means or DBSCAN instead, or sample then apply hierarchical.",
            "keyPoints": [
                "O(n²) space for distance matrix",
                "O(n² log n) time complexity",
                "Not scalable beyond ~10,000 samples",
                "Can use sampling, preprocessing, or alternative algorithms",
                "Best for small-medium datasets when hierarchy is valuable"
            ]
        }
    ],
    "dbscan-density-clustering": [
        {
            "question": "Explain how DBSCAN identifies clusters without requiring the number of clusters K as input. What are core points, border points, and noise points, and how are they determined?",
            "hint": "Cover the epsilon and MinPts parameters and how they define density.",
            "sampleAnswer": "DBSCAN discovers clusters as high-density regions separated by low-density regions, without needing K upfront. PARAMETERS: Epsilon (ε) = neighborhood radius; MinPts = minimum points to form dense region. POINT TYPES: (1) CORE POINT: has ≥ MinPts neighbors within ε distance (including itself). Forms cluster backbone. (2) BORDER POINT: has < MinPts neighbors but is within ε of a core point. Belongs to cluster but doesn't expand it. (3) NOISE POINT: neither core nor border. In low-density regions, marked as outliers. ALGORITHM: (1) For each unvisited point, count neighbors within ε, (2) If ≥ MinPts, start new cluster and recursively add all density-connected points, (3) If < MinPts but near core point, mark as border of that cluster, (4) Otherwise mark as noise. NUMBER OF CLUSTERS: determined by data density structure, not predefined. A point is density-reachable from another if connected through core points. This allows arbitrary cluster shapes unlike K-Means.",
            "keyPoints": [
                "Discovers clusters based on density, not predefined K",
                "Core points: ≥ MinPts neighbors within ε",
                "Border points: in ε-neighborhood of core point",
                "Noise points: neither core nor border, marked as outliers",
                "Number of clusters determined by data structure"
            ]
        },
        {
            "question": "How do you choose appropriate values for epsilon (ε) and MinPts in DBSCAN? Explain the k-distance graph method and provide practical guidelines.",
            "hint": "Cover the elbow in k-distance plot and rules of thumb for MinPts.",
            "sampleAnswer": "CHOOSING MinPts: RULE OF THUMB: MinPts ≥ dimensions + 1. For 2D: MinPts ≥ 3. Common values: 4, 5, 10. Increase for noisy data (more strict), decrease for sparse data. Start with MinPts = 5 as default. CHOOSING EPSILON: Use K-DISTANCE GRAPH: (1) For each point, compute distance to k-th nearest neighbor (k = MinPts), (2) Sort distances ascending, (3) Plot sorted distances, (4) Look for 'elbow' - where curve sharply increases, (5) Points before elbow are in dense regions, after are outliers. Elbow height suggests good ε. INTUITION: Below ε, points in clusters; above ε, in sparse regions. PRACTICAL TIPS: (1) Scale features first (DBSCAN uses distance), (2) Try multiple ε values around elbow, (3) Use silhouette score to compare, (4) Domain knowledge: what distance makes sense?, (5) Start with ε that gives reasonable number of clusters (2-10). PROBLEM: Single ε struggles with varying density - consider HDBSCAN for adaptive ε.",
            "keyPoints": [
                "MinPts ≥ dimensions + 1, typically 4-10",
                "Use k-distance graph to find epsilon",
                "Elbow in sorted k-distances suggests good ε",
                "Must scale features (distance-based)",
                "Single ε struggles with varying density"
            ]
        },
        {
            "question": "Compare DBSCAN with K-Means and Hierarchical clustering. In what scenarios does DBSCAN excel, and when should you use alternatives instead?",
            "hint": "Consider cluster shapes, scalability, parameter requirements, and handling of outliers.",
            "sampleAnswer": "DBSCAN ADVANTAGES: (1) ARBITRARY SHAPES: finds non-spherical clusters (moons, spirals) that K-Means misses, (2) OUTLIER DETECTION: explicitly identifies noise points, (3) NO K REQUIRED: number of clusters determined automatically, (4) ROBUST: insensitive to initialization (deterministic). LIMITATIONS: (1) PARAMETER SENSITIVITY: ε and MinPts difficult to tune, (2) VARYING DENSITY: single ε cannot handle clusters of different densities, (3) HIGH DIMENSIONS: curse of dimensionality makes distances meaningless, (4) BORDER POINTS: assignment can be ambiguous. WHEN TO USE DBSCAN: Geospatial data, arbitrary-shaped clusters, need outlier detection, clusters have similar density. USE K-MEANS: Spherical clusters, fast result needed, very large datasets, K known. USE HIERARCHICAL: Small dataset, need dendrogram, K unknown, varying density OK. USE HDBSCAN: DBSCAN benefits + varying density. PERFORMANCE: DBSCAN O(n log n) with spatial index vs K-Means O(nKt) - DBSCAN can be faster if K large.",
            "keyPoints": [
                "DBSCAN: arbitrary shapes, outlier detection, no K needed",
                "K-Means: faster, spherical clusters, requires K",
                "Hierarchical: dendrogram, no K, but slow",
                "DBSCAN excels with non-spherical clusters and noise",
                "Use HDBSCAN for varying density"
            ]
        }
    ],
    "principal-component-analysis": [
        {
            "question": "Explain the mathematical foundation of PCA. How does PCA use eigenvalue decomposition of the covariance matrix to find principal components, and what do eigenvalues and eigenvectors represent?",
            "hint": "Cover covariance matrix, eigendecomposition, and the meaning of eigenvalues/eigenvectors.",
            "sampleAnswer": "PCA finds orthogonal axes that maximize variance. MATHEMATICAL STEPS: (1) CENTER DATA: X_centered = X - mean, (2) COVARIANCE MATRIX: Σ = (1/n-1) X_centered^T @ X_centered. Measures how features vary together. (3) EIGENDECOMPOSITION: Σv = λv. Finds eigenvectors (v) and eigenvalues (λ). (4) SORT: order eigenvectors by eigenvalues (descending). (5) PROJECT: X_pca = X_centered @ V_k (top k eigenvectors). INTERPRETATION: EIGENVECTORS (principal components) = new axes, directions of maximum variance. Orthogonal to each other. Linear combinations of original features. EIGENVALUES = variance along each eigenvector. Large eigenvalue = important direction. Sum of eigenvalues = total variance. Eigenvalue / sum(eigenvalues) = proportion of variance explained. PCA rotates coordinate system to align with data's main variation. PC1 points in direction of maximum variance, PC2 in direction of maximum remaining variance orthogonal to PC1, etc.",
            "keyPoints": [
                "Covariance matrix captures feature relationships",
                "Eigenvectors are new axes (principal components)",
                "Eigenvalues represent variance along each axis",
                "Components ordered by decreasing eigenvalue",
                "Projects data onto directions of maximum variance"
            ]
        },
        {
            "question": "How do you choose the number of principal components to retain? Explain the explained variance ratio and compare different selection methods (elbow, 95% threshold, cross-validation).",
            "hint": "Cover cumulative explained variance and trade-offs between information retention and dimensionality reduction.",
            "sampleAnswer": "EXPLAINED VARIANCE RATIO: Each PC's eigenvalue / total variance. Indicates importance. Cumulative sum shows total variance retained. SELECTION METHODS: (1) THRESHOLD: Keep PCs explaining 95% (or 99%) of variance. Most common. Ensures minimal information loss. (2) ELBOW METHOD: Plot explained variance vs component number. Keep components before elbow (where variance drops sharply). (3) KAISER CRITERION: Keep PCs with eigenvalue > 1 (for standardized data). Indicates PC captures more variance than single original feature. (4) CROSS-VALIDATION: Try different k, evaluate on downstream task (classification, regression). Choose k maximizing task performance. (5) INTERPRETATION: Keep components you can interpret based on loadings. TRADE-OFFS: More PCs = more information but less reduction. Fewer PCs = more reduction but information loss. PRACTICAL: Start with 95% threshold. For visualization, use 2-3 PCs regardless of variance. For preprocessing, use CV on downstream task. RULE: Always check cumulative variance plot - shows diminishing returns of additional PCs.",
            "keyPoints": [
                "Explained variance ratio = eigenvalue / total variance",
                "Common: retain PCs explaining 95-99% variance",
                "Elbow method looks for drop-off point",
                "Cross-validation chooses based on task performance",
                "Trade-off between information retention and dimensionality"
            ]
        },
        {
            "question": "Explain how to interpret principal components through loadings. How do loadings help understand what each PC represents, and why is this important for feature engineering?",
            "hint": "Cover what loadings are, how to create biplots, and examples of interpretation.",
            "sampleAnswer": "LOADINGS: Correlations between original features and PCs. High loading = feature strongly contributes to PC. CALCULATION: Loading_ij = eigenvector_ij × sqrt(eigenvalue_j). Shows how much original feature i contributes to PC j. INTERPRETATION: PC1 loadings show which features vary together most. Positive loading = feature increases with PC. Negative loading = feature decreases with PC. EXAMPLE: If PC1 has high positive loadings for [height, weight, shoe_size], PC1 represents 'body size'. If PC2 has positive loading for age, negative for elasticity, PC2 represents 'aging'. BIPLOT: Plots data points in PC space AND loading vectors. Arrows show original features. Long arrow = feature important for those PCs. Parallel arrows = correlated features. IMPORTANCE FOR FEATURE ENGINEERING: (1) Identify correlated feature groups, (2) Understand data structure, (3) Create interpretable combinations, (4) Identify redundant features (high loading on same PC), (5) Domain validation: do PCs make sense? WARNING: PCs are linear combinations - harder to interpret than original features. Trade-off: reduced dimensionality vs interpretability.",
            "keyPoints": [
                "Loadings show correlation between features and PCs",
                "Help interpret what each PC represents",
                "Biplot visualizes both data and loadings",
                "Reveals correlated features and redundancy",
                "Trade-off: dimensionality reduction vs interpretability"
            ]
        }
    ],
    "other-dimensionality-reduction": [
        {
            "question": "Compare t-SNE and UMAP for dimensionality reduction. What are the key algorithmic differences, and in what scenarios would you choose one over the other?",
            "hint": "Cover computational efficiency, global vs local structure, and the ability to transform new data.",
            "sampleAnswer": "t-SNE: Converts high-D and low-D distances to probabilities, minimizes KL divergence. Preserves LOCAL structure (neighborhoods). UMAP: Based on Riemannian geometry and topological data analysis. Preserves BOTH local and global structure. KEY DIFFERENCES: (1) SPEED: UMAP much faster (can handle millions of points vs thousands for t-SNE), (2) GLOBAL STRUCTURE: UMAP preserves, t-SNE doesn't. UMAP better shows relationships between clusters. (3) NEW DATA: UMAP has .transform() method for new points. t-SNE must re-run entire algorithm. (4) STABILITY: UMAP more stable across runs. t-SNE very sensitive to random initialization. (5) PARAMETERS: t-SNE (perplexity, iterations). UMAP (n_neighbors, min_dist). WHEN TO USE t-SNE: Publication-quality visualizations, small datasets (<10K), only care about local structure. WHEN TO USE UMAP: Large datasets, need to transform new data, want global structure, general-purpose use. PRACTICAL: Start with UMAP for most use cases. Use t-SNE for beautiful visualizations of small datasets. Both better than PCA for non-linear structure.",
            "keyPoints": [
                "t-SNE: preserves local structure, slow, no transform for new data",
                "UMAP: preserves both local and global structure, fast",
                "UMAP can transform new data, t-SNE cannot",
                "UMAP more scalable (millions vs thousands)",
                "t-SNE best for visualization; UMAP for general use"
            ]
        },
        {
            "question": "What do t-SNE and UMAP visualizations tell us, and what don't they tell us? Explain common misinterpretations and best practices for interpreting these dimensionality reduction plots.",
            "hint": "Cover cluster sizes, inter-cluster distances, and the limitations of 2D projections.",
            "sampleAnswer": "WHAT THEY SHOW: (1) Which points are similar (close in plot = similar in high-D), (2) Rough cluster structure (groups that exist), (3) Relative neighborhood relationships. WHAT THEY DON'T SHOW (CRITICAL): (1) CLUSTER SIZE: Expansion/contraction is arbitrary. Large cluster in plot doesn't mean more points or higher density. (2) INTER-CLUSTER DISTANCE: Distance between clusters is meaningless. Two clusters close in plot may be far in high-D, or vice versa. (3) ABSOLUTE DISTANCES: Only local neighborhoods preserved. (4) AXES: No interpretable meaning (unlike PCA components). COMMON MISINTERPRETATIONS: 'Cluster A is bigger/denser than B' (No!), 'Clusters A and B are more similar than A and C' (No!), 'These two subclusters should merge' (Not enough info). BEST PRACTICES: (1) Run with multiple random seeds (especially t-SNE), (2) Try different perplexity/n_neighbors, (3) Use color to show known labels if available, (4) Validate clusters with domain knowledge, (5) Combine with other methods (hierarchical, silhouette scores), (6) Don't make quantitative claims from visualization alone. These are VISUALIZATION tools, not analysis tools. Use for exploration, not conclusions.",
            "keyPoints": [
                "Show: which points are similar, rough cluster structure",
                "Don't show: cluster sizes, inter-cluster distances, absolute distances",
                "Common error: interpreting cluster size or distance between clusters",
                "Best practice: try multiple parameters, validate with other methods",
                "Visualization tools, not definitive analysis"
            ]
        },
        {
            "question": "Explain the concept of manifold learning. How do techniques like Isomap, LLE, and t-SNE/UMAP assume data lies on a low-dimensional manifold embedded in high-dimensional space?",
            "hint": "Cover the manifold hypothesis and how different algorithms exploit it.",
            "sampleAnswer": "MANIFOLD HYPOTHESIS: High-dimensional data often lies on or near a low-dimensional manifold (smooth surface) embedded in the high-D space. EXAMPLE: Images of a face rotated 360° appear high-D (pixel space) but actually lie on a 2D manifold (rotation parameters). MANIFOLD LEARNING: Algorithms that discover and represent this manifold structure. APPROACHES: (1) ISOMAP: Uses geodesic distances (distances along manifold surface, not Euclidean). Builds nearest-neighbor graph, computes shortest paths, applies MDS. Assumes single connected manifold. Good for: curved surfaces like swiss roll. (2) LLE (Locally Linear Embedding): Assumes manifold locally linear. Represents each point as weighted combination of neighbors, preserves these weights in low-D. Good for: smooth manifolds. (3) t-SNE/UMAP: Model probability of points being neighbors in high-D and low-D, minimize divergence. Don't explicitly model manifold but implicitly preserve manifold structure. WHY IT WORKS: Real data has structure - not random points in high-D space. Correlations and constraints create lower intrinsic dimensionality. Face images: determined by face parameters (pose, expression), not independent pixels. Manifold learning reveals true degrees of freedom.",
            "keyPoints": [
                "High-D data often lies on low-D manifold",
                "Isomap: geodesic distances along manifold surface",
                "LLE: locally linear structure preservation",
                "t-SNE/UMAP: probabilistic neighborhood modeling",
                "Exploits data structure to find true dimensionality"
            ]
        }
    ],
    "anomaly-detection": [
        {
            "question": "Compare statistical methods (Z-score, IQR) with machine learning methods (Isolation Forest, LOF) for anomaly detection. What are the assumptions, advantages, and limitations of each approach?",
            "hint": "Consider distributional assumptions, multivariate vs univariate, and scalability.",
            "sampleAnswer": "STATISTICAL METHODS: Z-SCORE: Assumes normal distribution. Identifies points > 3 standard deviations from mean. Pros: simple, interpretable, fast. Cons: assumes normality, univariate (checks each feature independently), misses multivariate outliers (normal on each feature but unusual combination). IQR: Uses quartiles (Q1, Q3). Outliers outside [Q1-1.5×IQR, Q3+1.5×IQR]. Pros: robust to distribution, doesn't assume normality. Cons: still univariate, arbitrary threshold. ML METHODS: ISOLATION FOREST: Tree-based. Anomalies are easier to isolate (fewer splits). Pros: handles multivariate, no distribution assumption, fast, works in high-D. Cons: parameters (contamination) hard to set. LOF (Local Outlier Factor): Density-based. Compares point's density to neighbors'. Pros: finds local anomalies, handles varying density. Cons: slow (O(n²)), sensitive to parameters. WHEN TO USE: Statistical: quick baseline, univariate, normal data. Isolation Forest: general-purpose, high-D, large datasets. LOF: local anomalies, varying density, smaller datasets. BEST PRACTICE: Start with Isolation Forest for most cases. Combine methods for robustness.",
            "keyPoints": [
                "Statistical: simple, univariate, assumes distribution",
                "Z-score assumes normality; IQR more robust",
                "Isolation Forest: multivariate, no assumptions, scalable",
                "LOF: local anomalies, handles varying density, slower",
                "Isolation Forest best for general-purpose use"
            ]
        },
        {
            "question": "Explain the Isolation Forest algorithm. Why are anomalies easier to isolate than normal points, and how does this translate to shorter path lengths in isolation trees?",
            "hint": "Cover the intuition behind isolation, tree construction, and path length scoring.",
            "sampleAnswer": "CORE INTUITION: Anomalies are 'few and different' - they're rare and far from normal points. This makes them easier to separate (isolate) from the rest of the data. ALGORITHM: (1) Build many random trees (forest). Each tree: recursively split data on random features at random thresholds until points isolated. (2) Record path length for each point: number of splits from root to leaf. (3) Anomaly score: based on average path length across all trees. Short path = anomaly. WHY ANOMALIES HAVE SHORT PATHS: Consider splitting randomly: NORMAL POINT in dense region requires many splits to separate from neighbors (long path). ANOMALY in sparse region gets isolated quickly with few splits (short path). ANALOGY: Finding a lone person in empty field (easy, few steps) vs finding specific person in crowd (hard, many steps). MATHEMATICAL: Expected path length E(h(x)) for normal points ≈ average binary tree height ≈ 2(log n - 1). Anomalies have h(x) << log n. ANOMALY SCORE: Normalized: s(x) = 2^(-E(h(x))/c) where c normalizes. s ≈ 1: anomaly, s ≈ 0.5: normal, s < 0.5: safe.",
            "keyPoints": [
                "Anomalies are few and different, easier to isolate",
                "Random trees: split on random features and values",
                "Path length: splits needed to isolate point",
                "Normal points in dense regions: long paths",
                "Anomalies in sparse regions: short paths"
            ]
        },
        {
            "question": "In anomaly detection, what is the contamination parameter and how do you set it? What happens if you set it incorrectly, and how can you validate your anomaly detection results?",
            "hint": "Cover the precision-recall trade-off and validation strategies when labels are unavailable.",
            "sampleAnswer": "CONTAMINATION: Expected proportion of anomalies in dataset. Sets threshold for classification. contamination=0.1 means expect 10% anomalies. SETTING IT: (1) DOMAIN KNOWLEDGE: often best source. Fraud rate? Defect rate? (2) DATA EXPLORATION: plot anomaly scores, look for natural threshold. (3) BUSINESS REQUIREMENTS: cost of false positives vs false negatives. (4) START: 1-5% if unknown. INCORRECT SETTING: TOO HIGH (e.g., 0.2 when true 0.01): many false positives, normal points flagged, wastes investigation time. TOO LOW (e.g., 0.01 when true 0.1): miss many anomalies, false negatives, fail to catch issues. VALIDATION WITHOUT LABELS: (1) DOMAIN EXPERTS: manually review flagged anomalies. Do they make sense? (2) ANOMALY SCORE DISTRIBUTION: check for clear separation. (3) CONSISTENCY: do multiple methods agree? (4) TEMPORAL: do anomalies repeat or are they one-time events? (5) IMPACT: if acted upon, did flagged anomalies cause real problems? WITH LABELS (rare): use precision, recall, F1, PR curve. BEST PRACTICE: Set conservatively (low contamination), review flagged cases, adjust based on precision of actual anomalies.",
            "keyPoints": [
                "Contamination: expected proportion of anomalies",
                "Use domain knowledge or start with 1-5%",
                "Too high: many false positives; too low: miss anomalies",
                "Validate with domain experts and score distributions",
                "Monitor and adjust based on real-world performance"
            ]
        }
    ],
    "association-rule-learning": [
        {
            "question": "Explain the three key metrics in association rule mining: support, confidence, and lift. Why is lift more informative than confidence alone, and how do you interpret lift values?",
            "hint": "Cover the formulas and provide examples showing why confidence can be misleading.",
            "sampleAnswer": "SUPPORT: P(X) = how often itemset appears. support({milk, bread}) = 0.3 means 30% of transactions contain both. Indicates frequency/importance. CONFIDENCE: P(Y|X) = how often rule is true. confidence({milk}→{bread}) = P({milk,bread})/P({milk}). If 0.8, then 80% of customers who buy milk also buy bread. Problem: misleading if bread is very popular. LIFT: P(Y|X)/P(Y) = how much more likely Y is purchased with X vs without. lift({milk}→{bread}) = confidence/P({bread}). INTERPRETATION: Lift > 1: positive correlation (buying X increases chance of Y), Lift = 1: independent (X doesn't affect Y), Lift < 1: negative correlation (buying X decreases chance of Y). EXAMPLE: bread has P(bread)=0.9 (very popular). Rule {milk}→{bread} has confidence=0.9. Sounds strong! But lift=0.9/0.9=1.0. Actually independent - milk doesn't increase bread purchases. WHY LIFT MATTERS: Controls for base rate. Confidence alone misleading for popular items. Lift reveals true association. PRACTICAL: Focus on rules with lift > 1.2 (20% increase). Very high lift (>2) indicates strong association worth investigating.",
            "keyPoints": [
                "Support: frequency of itemset in data",
                "Confidence: conditional probability P(Y|X)",
                "Lift: confidence adjusted for base rate of Y",
                "Lift > 1: positive association; = 1: independent; < 1: negative",
                "Lift more informative than confidence alone"
            ]
        },
        {
            "question": "Explain the Apriori algorithm and the Apriori property. How does this property dramatically reduce the number of candidate itemsets that need to be checked, and why is this important for computational efficiency?",
            "hint": "Cover the pruning strategy and how it prevents combinatorial explosion.",
            "sampleAnswer": "APRIORI PROPERTY (Monotonicity): If an itemset is frequent, all its subsets must also be frequent. Contrapositive: If an itemset is infrequent, all its supersets must also be infrequent. IMPORTANCE: Massive computational savings. EXAMPLE: If {milk, bread} has support < threshold, then {milk, bread, eggs}, {milk, bread, butter}, etc., must also be infrequent. No need to check! ALGORITHM: (1) Find frequent 1-itemsets (scan database), (2) Generate candidate k-itemsets from frequent (k-1)-itemsets, (3) Prune candidates: if any (k-1)-subset is infrequent, remove candidate, (4) Count support of remaining candidates (scan database), (5) Keep frequent k-itemsets, (6) Repeat until no new frequent itemsets. WITHOUT PRUNING: With n items, must check 2^n possible itemsets. For n=100: 2^100 ≈ 10^30 itemsets - impossible! WITH PRUNING: If 95 items are infrequent individually, 2^95 supersets eliminated immediately. Practical: check thousands instead of billions. TRADE-OFF: Multiple database scans (slow) but dramatically fewer candidates. Alternative: FP-Growth avoids repeated scans using tree structure (faster for large databases).",
            "keyPoints": [
                "Apriori property: frequent itemset → all subsets frequent",
                "Allows pruning of candidate itemsets",
                "Prevents combinatorial explosion (2^n possibilities)",
                "Dramatically reduces computational cost",
                "Multiple database scans but far fewer candidates"
            ]
        },
        {
            "question": "Discuss practical applications and limitations of association rule mining. Why don't association rules imply causation, and what are common pitfalls in interpreting rules?",
            "hint": "Cover use cases, the causation issue, spurious correlations, and business implementation.",
            "sampleAnswer": "APPLICATIONS: (1) RETAIL: 'Customers who bought X also bought Y' recommendations, product placement (separate associated items to increase store traversal), bundling/promotions. (2) E-COMMERCE: product recommendations, frequently bought together. (3) HEALTHCARE: comorbidity analysis (diseases that co-occur), treatment protocols. (4) WEB: page navigation patterns, click analysis. CAUSATION WARNING: Association ≠ Causation! {diapers}→{beer} doesn't mean diapers cause beer purchases. Possible explanations: (1) common cause (new parents), (2) coincidence, (3) temporal patterns (both bought on weekends). Cannot determine causality from association rules. Need experiments (A/B tests) or causal inference methods. COMMON PITFALLS: (1) SPURIOUS CORRELATIONS: random chance in large datasets. (2) POPULAR ITEMS: dominate rules but aren't interesting. (3) TRIVIAL RULES: {bread, butter}→{milk} and {bread, milk}→{butter} are redundant. (4) STATIC: rules change over time (seasonality, trends). (5) ACTIONABILITY: interesting ≠ actionable. Need business context. BEST PRACTICES: Use lift > 1.2, validate with domain experts, remove trivial/redundant rules, update periodically, A/B test before implementing, consider confounding factors.",
            "keyPoints": [
                "Applications: recommendations, product placement, comorbidity analysis",
                "Association does not imply causation",
                "Watch for spurious correlations and trivial rules",
                "Rules are static, need periodic updates",
                "Validate with domain knowledge and A/B testing"
            ]
        }
    ]
}

# Write quiz files
for section_id, questions in quizzes.items():
    filename = f"lib/content/quizzes/ml-unsupervised-learning/{section_id}-quiz.ts"
    
    content = f'''/**
 * Quiz: {section_id.replace('-', ' ').title()}
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import {{ QuizQuestion }} from '../../../types';

export const {section_id.replace('-', '_')}Quiz: QuizQuestion[] = [
'''
    
    for i, q in enumerate(questions):
        # Format keyPoints array
        key_points_str = ',\\n    '.join([f"'{kp}'" for kp in q['keyPoints']])
        
        content += f'''  {{
    id: '{section_id}-q{i+1}',
    question: `{q['question']}`,
    hint: '{q['hint']}',
    sampleAnswer: `{q['sampleAnswer']}`,
    keyPoints: [
      {key_points_str}
    ]
  }}{',' if i < len(questions) - 1 else ''}
'''
    
    content += '];\n'
    
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"Created: {filename}")

print(f"\\nAll {len(quizzes)} quiz files created successfully!")
