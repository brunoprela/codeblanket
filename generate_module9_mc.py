#!/usr/bin/env python3
"""
Generate all Module 9 multiple choice files
"""

import os

# Create directory if needed
os.makedirs("lib/content/multiple-choice/ml-unsupervised-learning", exist_ok=True)

mc_questions = {
    "unsupervised-learning-overview": [
        {
            "question": "What is the primary difference between supervised and unsupervised learning?",
            "options": [
                "Supervised learning is faster than unsupervised learning",
                "Supervised learning requires labeled data, while unsupervised learning does not",
                "Unsupervised learning is more accurate than supervised learning",
                "Supervised learning works only with numerical data"
            ],
            "correctAnswer": 1,
            "explanation": "The fundamental difference is that supervised learning requires labeled training data (input-output pairs), while unsupervised learning discovers patterns in unlabeled data without predefined target variables."
        },
        {
            "question": "Which of the following is NOT a main category of unsupervised learning?",
            "options": [
                "Clustering",
                "Classification",
                "Dimensionality Reduction",
                "Anomaly Detection"
            ],
            "correctAnswer": 1,
            "explanation": "Classification is a supervised learning task that requires labeled data. The three main categories of unsupervised learning are clustering, dimensionality reduction, and anomaly detection."
        },
        {
            "question": "The curse of dimensionality refers to:",
            "options": [
                "The difficulty of storing high-dimensional data",
                "The phenomenon where data becomes sparse and distances less meaningful in high dimensions",
                "The inability to visualize data beyond 3 dimensions",
                "The requirement for more computational power"
            ],
            "correctAnswer": 1,
            "explanation": "The curse of dimensionality describes how, as dimensionality increases, data becomes increasingly sparse, distances between points become less meaningful, and algorithms become less effective. This fundamentally affects how unsupervised learning algorithms work."
        }
    ],
    "k-means-clustering": [
        {
            "question": "In K-Means clustering, what does the algorithm minimize?",
            "options": [
                "The maximum distance between any two points",
                "The within-cluster sum of squares (WCSS)",
                "The number of iterations needed for convergence",
                "The distance between cluster centroids"
            ],
            "correctAnswer": 1,
            "explanation": "K-Means aims to minimize the Within-Cluster Sum of Squares (WCSS), also called inertia - the sum of squared distances from each point to its assigned cluster centroid. This creates compact, spherical clusters."
        },
        {
            "question": "What is the main advantage of K-Means++ initialization over random initialization?",
            "options": [
                "It requires fewer clusters",
                "It runs faster",
                "It chooses initial centroids that are spread apart, leading to better and faster convergence",
                "It can find non-spherical clusters"
            ],
            "correctAnswer": 2,
            "explanation": "K-Means++ uses a probabilistic method to select initial centroids that are far apart from each other, which leads to better final results and faster convergence compared to random initialization."
        },
        {
            "question": "Which of the following is a limitation of K-Means?",
            "options": [
                "It works only with categorical data",
                "It assumes clusters are spherical and of similar size",
                "It cannot handle more than 10 features",
                "It requires knowing the data labels"
            ],
            "correctAnswer": 1,
            "explanation": "K-Means assumes clusters are roughly spherical (convex), of similar size, and have similar density. It struggles with elongated, non-convex shapes (like crescent moons) and clusters of very different sizes."
        },
        {
            "question": "What does the elbow method help determine in K-Means?",
            "options": [
                "The optimal learning rate",
                "The optimal number of clusters (K)",
                "The optimal number of iterations",
                "The optimal distance metric"
            ],
            "correctAnswer": 1,
            "explanation": "The elbow method plots WCSS vs K and looks for the 'elbow' point where adding more clusters yields diminishing returns. This suggests the optimal number of clusters."
        }
    ],
    "hierarchical-clustering": [
        {
            "question": "What is the time complexity of hierarchical clustering?",
            "options": [
                "O(n)",
                "O(n log n)",
                "O(n² log n)",
                "O(n³)"
            ],
            "correctAnswer": 2,
            "explanation": "Hierarchical clustering has O(n² log n) time complexity with optimizations (naive implementation is O(n³)), and O(n²) space complexity for the distance matrix. This makes it impractical for very large datasets."
        },
        {
            "question": "In hierarchical clustering, Ward linkage minimizes:",
            "options": [
                "The minimum distance between clusters",
                "The maximum distance between clusters",
                "The within-cluster variance",
                "The number of merges needed"
            ],
            "correctAnswer": 2,
            "explanation": "Ward linkage merges clusters to minimize the increase in within-cluster variance, similar to the K-Means objective. This tends to create compact, spherical clusters of similar size."
        },
        {
            "question": "What advantage does hierarchical clustering have over K-Means?",
            "options": [
                "It's faster for large datasets",
                "It doesn't require specifying the number of clusters upfront",
                "It can only find spherical clusters",
                "It uses less memory"
            ],
            "correctAnswer": 1,
            "explanation": "Hierarchical clustering creates a dendrogram showing relationships at all levels, allowing you to choose the number of clusters after seeing the hierarchy. K-Means requires specifying K before running the algorithm."
        },
        {
            "question": "In a dendrogram, what does the height of a merge represent?",
            "options": [
                "The number of points in the cluster",
                "The distance or dissimilarity between the clusters being merged",
                "The order in which clusters were formed",
                "The variance within the cluster"
            ],
            "correctAnswer": 1,
            "explanation": "The height (y-axis) of a merge in a dendrogram represents the distance or dissimilarity between the clusters being merged. Larger heights indicate merging more dissimilar clusters, suggesting a good place to cut."
        }
    ],
    "dbscan-density-clustering": [
        {
            "question": "In DBSCAN, a core point is defined as:",
            "options": [
                "A point at the center of a cluster",
                "A point with at least MinPts neighbors within epsilon distance",
                "A point that is an outlier",
                "The first point visited by the algorithm"
            ],
            "correctAnswer": 1,
            "explanation": "A core point in DBSCAN has at least MinPts neighbors (including itself) within epsilon (ε) distance. Core points form the backbone of clusters and can extend the cluster to other density-connected points."
        },
        {
            "question": "What is a key advantage of DBSCAN over K-Means?",
            "options": [
                "DBSCAN is always faster",
                "DBSCAN can find arbitrarily shaped clusters and identify outliers",
                "DBSCAN requires less memory",
                "DBSCAN works better with very high-dimensional data"
            ],
            "correctAnswer": 1,
            "explanation": "DBSCAN can discover clusters of arbitrary shapes (not just spherical) and explicitly identifies outliers as noise points, unlike K-Means which forces every point into a cluster and assumes spherical shapes."
        },
        {
            "question": "How do you choose an appropriate epsilon (ε) value for DBSCAN?",
            "options": [
                "Always use ε = 1.0",
                "Use the k-distance graph and look for the elbow point",
                "Set ε equal to the number of clusters desired",
                "Use the average distance between all points"
            ],
            "correctAnswer": 1,
            "explanation": "The k-distance graph method plots sorted distances to the k-th nearest neighbor (k = MinPts). The 'elbow' point in this curve suggests a good epsilon value, separating dense regions from sparse regions."
        },
        {
            "question": "What is a major limitation of DBSCAN?",
            "options": [
                "It cannot detect outliers",
                "It requires knowing K in advance",
                "It struggles with clusters of varying densities",
                "It only works with categorical data"
            ],
            "correctAnswer": 2,
            "explanation": "DBSCAN uses a single epsilon value, which makes it difficult to handle clusters with significantly different densities. HDBSCAN (Hierarchical DBSCAN) addresses this limitation by using varying epsilon values."
        }
    ],
    "principal-component-analysis": [
        {
            "question": "What do the principal components in PCA represent?",
            "options": [
                "The original features in descending order of importance",
                "New orthogonal axes that maximize variance",
                "The cluster centers of the data",
                "The outliers in the dataset"
            ],
            "correctAnswer": 1,
            "explanation": "Principal components are new orthogonal (perpendicular) axes that are linear combinations of original features. They are ordered by the amount of variance they explain, with PC1 explaining the most variance."
        },
        {
            "question": "Why is feature scaling (standardization) crucial before applying PCA?",
            "options": [
                "PCA doesn't work on unscaled data",
                "To make the algorithm run faster",
                "To prevent features with larger scales from dominating the principal components",
                "To ensure all eigenvalues are positive"
            ],
            "correctAnswer": 2,
            "explanation": "PCA is sensitive to feature scales because it uses distances/variance. Features with larger scales (e.g., salary in dollars vs age in years) will dominate the principal components if not scaled, leading to biased results."
        },
        {
            "question": "What information does the explained variance ratio provide?",
            "options": [
                "The percentage of outliers in each component",
                "The proportion of total variance captured by each principal component",
                "The correlation between original features",
                "The number of samples in each cluster"
            ],
            "correctAnswer": 1,
            "explanation": "Explained variance ratio shows what percentage of the total variance in the data is captured by each principal component. This helps determine how many components to retain for dimensionality reduction."
        },
        {
            "question": "When should you NOT use PCA for dimensionality reduction?",
            "options": [
                "When features are highly correlated",
                "When you need interpretable features",
                "When you have high-dimensional data",
                "When you want to visualize data"
            ],
            "correctAnswer": 1,
            "explanation": "Avoid PCA when interpretability is crucial, as principal components are linear combinations of original features and lose direct interpretability. Also avoid it when relationships are non-linear (use Kernel PCA or t-SNE instead)."
        }
    ],
    "other-dimensionality-reduction": [
        {
            "question": "What is the main difference between PCA and t-SNE?",
            "options": [
                "PCA is supervised, t-SNE is unsupervised",
                "PCA is linear, t-SNE is non-linear",
                "PCA is slower than t-SNE",
                "PCA requires more parameters than t-SNE"
            ],
            "correctAnswer": 1,
            "explanation": "PCA is a linear dimensionality reduction technique that finds linear combinations of features, while t-SNE is non-linear and can capture complex manifold structures, making it better for visualization of complex data."
        },
        {
            "question": "Which statement about t-SNE visualizations is TRUE?",
            "options": [
                "The size of clusters indicates the number of points in each cluster",
                "The distance between clusters is meaningful and can be interpreted",
                "Points close together in the plot are similar in the high-dimensional space",
                "The axes have interpretable meanings like in PCA"
            ],
            "correctAnswer": 2,
            "explanation": "In t-SNE, points close together ARE similar in high-D space (local structure preserved). However, cluster sizes, inter-cluster distances, and axes are NOT meaningful or interpretable."
        },
        {
            "question": "What is a key advantage of UMAP over t-SNE?",
            "options": [
                "UMAP is simpler to understand",
                "UMAP preserves global structure better and can transform new data",
                "UMAP requires no parameters",
                "UMAP works only on small datasets"
            ],
            "correctAnswer": 1,
            "explanation": "UMAP preserves both local and global structure (unlike t-SNE which focuses on local), is faster, more scalable, and crucially has a .transform() method to project new data points without retraining."
        },
        {
            "question": "The manifold hypothesis in machine learning states that:",
            "options": [
                "All data is linearly separable",
                "High-dimensional data often lies on or near a low-dimensional manifold",
                "Dimensionality reduction always improves model performance",
                "Data must be normalized before analysis"
            ],
            "correctAnswer": 1,
            "explanation": "The manifold hypothesis suggests that real-world high-dimensional data often lies on or near a much lower-dimensional manifold (smooth surface) embedded in the high-dimensional space, which justifies dimensionality reduction."
        }
    ],
    "anomaly-detection": [
        {
            "question": "What is the main principle behind Isolation Forest?",
            "options": [
                "Anomalies form separate clusters",
                "Anomalies are easier to isolate and require fewer splits in random trees",
                "Anomalies have higher density than normal points",
                "Anomalies are always at the edges of the feature space"
            ],
            "correctAnswer": 1,
            "explanation": "Isolation Forest is based on the principle that anomalies are 'few and different', making them easier to isolate with fewer random splits in a tree. Normal points in dense regions require more splits to isolate."
        },
        {
            "question": "In Local Outlier Factor (LOF), an anomaly is characterized by:",
            "options": [
                "Being far from all other points",
                "Having lower density than its neighbors",
                "Being in the smallest cluster",
                "Having the highest feature values"
            ],
            "correctAnswer": 1,
            "explanation": "LOF identifies anomalies based on local density. A point is an outlier if its density is significantly lower than the density of its neighbors, making it effective for detecting local anomalies in varying density data."
        },
        {
            "question": "What does the contamination parameter in anomaly detection specify?",
            "options": [
                "The percentage of features that are corrupted",
                "The expected proportion of anomalies in the dataset",
                "The threshold distance for classifying outliers",
                "The number of anomaly detection methods to use"
            ],
            "correctAnswer": 1,
            "explanation": "The contamination parameter specifies the expected proportion (percentage) of anomalies in the dataset. It sets the threshold for classifying points as anomalies, typically ranging from 0.01 to 0.1 (1% to 10%)."
        },
        {
            "question": "Why is accuracy NOT a good metric for evaluating anomaly detection?",
            "options": [
                "Accuracy is too difficult to calculate",
                "Anomalies are rare, so high accuracy can be achieved by labeling everything as normal",
                "Accuracy only works for supervised learning",
                "Accuracy doesn't account for distance measures"
            ],
            "correctAnswer": 1,
            "explanation": "Due to class imbalance (anomalies are rare, often <1%), a naive classifier predicting 'normal' for everything achieves >99% accuracy but misses all anomalies. Use precision, recall, F1, or PR-AUC instead."
        }
    ],
    "association-rule-learning": [
        {
            "question": "In association rule mining, what does the support metric measure?",
            "options": [
                "How reliable the rule is",
                "How frequently an itemset appears in the dataset",
                "How much better than random the rule is",
                "The strength of the relationship between items"
            ],
            "correctAnswer": 1,
            "explanation": "Support measures how frequently an itemset appears in the dataset. For example, support({milk, bread}) = 0.3 means 30% of all transactions contain both milk and bread."
        },
        {
            "question": "Why is lift more informative than confidence alone?",
            "options": [
                "Lift is easier to calculate",
                "Lift adjusts for the base rate of the consequent item",
                "Lift can only be positive",
                "Lift works with continuous data"
            ],
            "correctAnswer": 1,
            "explanation": "Lift = confidence / P(Y) adjusts for how common Y is. A rule might have high confidence simply because Y is popular. Lift > 1 indicates X truly increases the likelihood of Y beyond its base rate."
        },
        {
            "question": "The Apriori property states that:",
            "options": [
                "If an itemset is frequent, all its subsets must be frequent",
                "Frequent items always appear together",
                "The algorithm must start with 1-itemsets",
                "Support decreases as itemset size increases"
            ],
            "correctAnswer": 0,
            "explanation": "The Apriori property (monotonicity) states that if an itemset is frequent, all its subsets must also be frequent. This allows pruning of candidates: if {A,B} is infrequent, no need to check {A,B,C}."
        },
        {
            "question": "What does a lift value of 0.8 for a rule {A} → {B} indicate?",
            "options": [
                "B is 80% likely when A is purchased",
                "80% of transactions contain both A and B",
                "Purchasing A decreases the likelihood of purchasing B",
                "The rule has 80% accuracy"
            ],
            "correctAnswer": 2,
            "explanation": "Lift < 1 indicates negative correlation: purchasing A actually makes B less likely than its base rate. Lift = 1 is independence, lift > 1 is positive correlation."
        }
    ]
}

# Write multiple choice files
for section_id, questions in mc_questions.items():
    filename = f"lib/content/multiple-choice/ml-unsupervised-learning/{section_id}-multiple-choice.ts"
    
    content = f'''/**
 * Multiple Choice Questions: {section_id.replace('-', ' ').title()}
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import {{ MultipleChoiceQuestion }} from '../../../types';

export const {section_id.replace('-', '_')}MultipleChoice: MultipleChoiceQuestion[] = [
'''
    
    for i, q in enumerate(questions):
        # Format options array
        options_str = ',\\n    '.join([f"'{opt}'" for opt in q['options']])
        
        content += f'''  {{
    id: '{section_id}-mc{i+1}',
    question: '{q['question']}',
    options: [
      {options_str}
    ],
    correctAnswer: {q['correctAnswer']},
    explanation: '{q['explanation']}'
  }}{',' if i < len(questions) - 1 else ''}
'''
    
    content += '];\n'
    
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"Created: {filename}")

print(f"\\nAll {len(mc_questions)} multiple choice files created successfully!")
