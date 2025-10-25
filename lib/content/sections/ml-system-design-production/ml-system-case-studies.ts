export const mlSystemCaseStudies = {
  title: 'ML System Case Studies',
  id: 'ml-system-case-studies',
  content: `
# ML System Case Studies

## Introduction

**"Theory is knowing how, practice is knowing when."**

Real-world ML systems face challenges that textbooks don't cover. This section examines production ML architectures from leading companies and trading systems to extract practical lessons.

**Case Studies Covered**:
1. Netflix Recommendation System
2. Uber\'s ML Platform (Michelangelo)
3. Algorithmic Trading System
4. Fraud Detection at Scale
5. Real-Time Bidding System

Each case study covers:
- Problem & constraints
- Architecture design
- Technical challenges
- Solutions & trade-offs
- Key learnings

---

## Case Study 1: Netflix Recommendation System

### Problem & Scale

\`\`\`python
"""
Netflix Recommendations at Scale
"""

netflix_stats = {
    "users": "230+ million",
    "daily_streams": "1+ billion",
    "catalog_size": "15,000+ titles",
    "regions": "190+ countries",
    "recommendation_percentage": "80% of viewing from recommendations",
    
    "challenges": [
        "Cold start problem (new users/content)",
        "Real-time personalization",
        "Diverse user preferences",
        "Global scale",
        "Content freshness"
    ]
}

print("=== Netflix Recommendation System ===\\n")
for key, value in netflix_stats.items():
    if key != 'challenges':
        print(f"{key}: {value}")

print("\\nKey Challenges:")
for challenge in netflix_stats['challenges']:
    print(f"  - {challenge}")
\`\`\`

### Architecture

\`\`\`python
"""
Netflix Recommendation Architecture
"""

class NetflixRecommendationArchitecture:
    """
    Simplified Netflix architecture
    
    Multiple algorithms working together:
    1. Collaborative Filtering (user-item interactions)
    2. Content-Based (metadata, genres)
    3. Ranking (personalized ordering)
    4. Page Generation (what to show where)
    """
    
    def __init__(self):
        self.architecture = {
            "offline_training": {
                "frequency": "Daily",
                "models": [
                    "Matrix Factorization (ALS)",
                    "Deep Neural Networks",
                    "XGBoost (CTR prediction)"
                ],
                "compute": "Spark on AWS EMR",
                "training_data": "100s of TB"
            },
            
            "online_serving": {
                "latency": "<100ms P99",
                "requests": "Millions per second",
                "infrastructure": "Microservices on AWS",
                "caching": "Multi-level (CDN, Redis)"
            },
            
            "experimentation": {
                "ab_tests": "1000s concurrent",
                "framework": "Custom A/B testing platform",
                "metrics": ["Engagement", "Retention", "Satisfaction"]
            },
            
            "data_pipeline": {
                "streaming": "Kafka for real-time events",
                "batch": "S3 + Spark for historical data",
                "feature_store": "Custom feature management"
            }
        }
    
    def explain_algorithm_stack (self):
        """
        Netflix uses ensemble of algorithms
        """
        print("\\n=== Netflix Algorithm Stack ===\\n")
        
        print("1. Candidate Generation:")
        print("   - Collaborative Filtering (similar users)")
        print("   - Content-Based (similar titles)")
        print("   - Popularity (trending)")
        print("   - Continues Watching (resume)")
        print("   → Generates ~1000 candidates per user")
        
        print("\\n2. Ranking:")
        print("   - Neural network predicts P(watch | user, title)")
        print("   - Factors: watch history, time of day, device")
        print("   → Ranks candidates by predicted engagement")
        
        print("\\n3. Page Generation:")
        print("   - Decides which rows to show")
        print("   - Optimizes for diversity and discovery")
        print("   - Personalizes artwork")
        print("   → Final personalized homepage")
    
    def key_learnings (self):
        """
        Key lessons from Netflix
        """
        print("\\n=== Key Learnings ===\\n")
        
        learnings = {
            "Ensemble": "Single model isn't enough - combine multiple algorithms",
            "A/B Testing": "Everything is tested - 1000s of experiments running",
            "Personalization": "From recommendations to artwork to thumbnails",
            "Cold Start": "Use content metadata + transfer learning for new titles",
            "Infrastructure": "Invest in experimentation platform, not just models",
            "Metrics": "Focus on engagement & retention, not just accuracy"
        }
        
        for key, value in learnings.items():
            print(f"✓ {key}: {value}")


# Run case study
netflix = NetflixRecommendationArchitecture()
netflix.explain_algorithm_stack()
netflix.key_learnings()
\`\`\`

---

## Case Study 2: Uber\'s Michelangelo

### ML Platform Architecture

\`\`\`python
"""
Uber's Michelangelo ML Platform
"""

class UberMichelangelo:
    """
    Uber's end-to-end ML platform
    
    Powers:
    - ETA prediction
    - Fraud detection
    - Demand forecasting
    - Dynamic pricing
    - Driver-rider matching
    """
    
    def __init__(self):
        self.platform_components = {
            "data_management": {
                "feature_store": "Centralized features across all models",
                "streaming": "Kafka for real-time features",
                "batch": "Hive/Spark for historical",
                "monitoring": "Data quality checks"
            },
            
            "model_training": {
                "frameworks": ["XGBoost", "TensorFlow", "PyTorch"],
                "distributed": "Horovod for distributed training",
                "hyperparameter_tuning": "Bayesian optimization",
                "experiments": "MLflow for tracking"
            },
            
            "model_deployment": {
                "serving": "Custom model server (Java/Python)",
                "latency": "P99 < 10ms for critical models",
                "scale": "Millions predictions/sec",
                "update": "Canary deployment"
            },
            
            "monitoring": {
                "metrics": "Prediction quality, latency, throughput",
                "alerts": "PagerDuty for critical issues",
                "dashboards": "Grafana for visualization",
                "retraining": "Automatic when drift detected"
            }
        }
    
    def explain_feature_store (self):
        """
        Centralized feature store is key innovation
        """
        print("\\n=== Uber Feature Store ===\\n")
        
        print("Problem:")
        print("  - 100s of ML models")
        print("  - Each team computing same features")
        print("  - Training-serving skew")
        print("  - No feature reuse")
        
        print("\\nSolution: Centralized Feature Store")
        print("  ✓ Compute once, use everywhere")
        print("  ✓ Same features for training & serving")
        print("  ✓ Feature discovery & sharing")
        print("  ✓ Monitoring & quality control")
        
        print("\\nFeatures:")
        print("  - User features: trips, ratings, behavior")
        print("  - Driver features: acceptance rate, ratings")
        print("  - Location features: traffic, weather, events")
        print("  - Time features: hour, day, seasonality")
    
    def eta_prediction_model (self):
        """
        ETA prediction: Critical for UX
        """
        print("\\n=== ETA Prediction ===\\n")
        
        print("Requirements:")
        print("  - Latency: <50ms")
        print("  - Accuracy: Within 5% of actual time")
        print("  - Scale: Every trip request")
        
        print("\\nApproach:")
        print("  1. Gradient Boosted Trees (XGBoost)")
        print("     - Fast inference")
        print("     - Handles complex interactions")
        
        print("\\n  2. Features:")
        print("     - Historical trips on route")
        print("     - Current traffic")
        print("     - Time of day / day of week")
        print("     - Weather conditions")
        print("     - Driver behavior")
        
        print("\\n  3. Training:")
        print("     - Daily retraining on past week")
        print("     - 10M+ trips per day")
        print("     - Distributed training on Spark")
        
        print("\\n  4. Serving:")
        print("     - Model cached in memory")
        print("     - Features pre-computed")
        print("     - P99 latency: 8ms")
    
    def key_takeaways (self):
        """
        Lessons from Uber
        """
        print("\\n=== Key Takeaways ===\\n")
        
        takeaways = [
            "Platform > Individual Models: Build infrastructure for 100s of models",
            "Feature Store: Centralize feature computation and serving",
            "Low Latency: ETA/matching models need <10ms inference",
            "Continuous Training: Retrain daily with fresh data",
            "Monitoring: Track data quality, not just model metrics",
            "Canary Deployments: Gradual rollout with automatic rollback"
        ]
        
        for i, takeaway in enumerate (takeaways, 1):
            print(f"{i}. {takeaway}")


# Run case study
uber = UberMichelangelo()
uber.explain_feature_store()
uber.eta_prediction_model()
uber.key_takeaways()
\`\`\`

---

## Case Study 3: Algorithmic Trading System

### High-Frequency Trading ML

\`\`\`python
"""
Production Algorithmic Trading System
"""

class AlgoTradingSystem:
    """
    ML-powered trading system
    
    Requirements:
    - Ultra-low latency (<10ms)
    - High reliability (99.99%)
    - Risk management
    - Regulatory compliance
    """
    
    def __init__(self):
        self.system_requirements = {
            "latency": {
                "data_ingestion": "<1ms",
                "feature_computation": "<2ms",
                "model_inference": "<5ms",
                "order_placement": "<2ms",
                "total_budget": "<10ms end-to-end"
            },
            
            "reliability": {
                "uptime": "99.99%",
                "data_quality": "Real-time validation",
                "failover": "Hot standby",
                "monitoring": "24/7 alerting"
            },
            
            "risk_management": {
                "position_limits": "Per symbol and total",
                "drawdown_limit": "15% max",
                "kill_switch": "Automatic shutdown on anomaly",
                "circuit_breakers": "Prevent runaway trading"
            }
        }
    
    def architecture (self):
        """
        Trading system architecture
        """
        print("\\n=== Trading System Architecture ===\\n")
        
        print("1. Data Layer:")
        print("   - WebSocket: Real-time market data")
        print("   - TCP/FIX: Exchange connections")
        print("   - Redis: Feature cache (hot data)")
        print("   - TimescaleDB: Historical tick data")
        
        print("\\n2. Feature Engineering:")
        print("   - Technical indicators (cached)")
        print("   - Order book features (real-time)")
        print("   - Market microstructure")
        print("   - Sentiment scores (pre-computed)")
        
        print("\\n3. Models:")
        print("   - XGBoost: Price direction (60% accuracy)")
        print("   - LSTM: Volatility forecasting")
        print("   - Reinforcement Learning: Execution")
        print("   - Ensemble: Combine predictions")
        
        print("\\n4. Execution:")
        print("   - Smart order router")
        print("   - VWAP/TWAP algorithms")
        print("   - Slippage minimization")
        print("   - FIX protocol to exchanges")
        
        print("\\n5. Risk Management:")
        print("   - Pre-trade checks (position limits)")
        print("   - Real-time P&L monitoring")
        print("   - Automatic position reduction")
        print("   - Daily VaR calculation")
    
    def latency_optimization (self):
        """
        How to achieve <10ms latency
        """
        print("\\n=== Latency Optimization ===\\n")
        
        optimizations = {
            "Model": [
                "XGBoost/LightGBM (not deep learning)",
                "Model quantization (INT8)",
                "Single-precision floats",
                "Batch size = 1 (no batching delay)"
            ],
            
            "Features": [
                "Pre-compute expensive features",
                "Cache in Redis (sub-ms access)",
                "Only 20-30 features (not 100+)",
                "Avoid database queries in hot path"
            ],
            
            "Infrastructure": [
                "Co-located servers near exchange",
                "C++ for critical path, Python for research",
                "Lock-free data structures",
                "Zero-copy message passing"
            ],
            
            "Network": [
                "Direct market data feeds",
                "Kernel bypass (DPDK)",
                "10Gb/40Gb NICs",
                "Optimized TCP stack"
            ]
        }
        
        for category, items in optimizations.items():
            print(f"{category}:")
            for item in items:
                print(f"  ✓ {item}")
            print()
    
    def risk_framework (self):
        """
        Risk management system
        """
        print("\\n=== Risk Management Framework ===\\n")
        
        print("Pre-Trade Checks:")
        print("  1. Position limit: Max $1M per symbol")
        print("  2. Total exposure: Max $10M")
        print("  3. Max order size: $100K")
        print("  4. Blacklist check: Restricted symbols")
        
        print("\\nReal-Time Monitoring:")
        print("  1. P&L tracking (tick-by-tick)")
        print("  2. Drawdown alerts (>5%, >10%, >15%)")
        print("  3. Position concentration")
        print("  4. Sharpe ratio (rolling)")
        
        print("\\nAutomatic Actions:")
        print("  1. Stop trading if drawdown >15%")
        print("  2. Reduce positions if volatility spikes")
        print("  3. Kill switch on data quality issues")
        print("  4. Circuit breaker on rapid losses")
    
    def lessons_learned (self):
        """
        Hard-learned lessons
        """
        print("\\n=== Lessons Learned ===\\n")
        
        lessons = [
            ("Latency Matters", "1ms difference = millions in P&L"),
            ("Simple Models Win", "XGBoost > Deep Learning in HFT"),
            ("Cache Everything", "Database queries kill latency"),
            ("Test in Paper", "Never go live without paper trading"),
            ("Monitor Constantly", "Alerts for everything"),
            ("Risk First", "Protecting capital > maximizing returns"),
            ("Data Quality", "Bad data = bad trades"),
            ("Failover Plans", "Always have backup systems")
        ]
        
        for title, lesson in lessons:
            print(f"✓ {title}: {lesson}")


# Run case study
trading = AlgoTradingSystem()
trading.architecture()
trading.latency_optimization()
trading.risk_framework()
trading.lessons_learned()
\`\`\`

---

## Case Study 4: Fraud Detection at Scale

### PayPal/Stripe Fraud Detection

\`\`\`python
"""
Real-Time Fraud Detection System
"""

class FraudDetectionSystem:
    """
    Fraud detection at payment processors
    
    Challenges:
    - Real-time (must approve/decline in <100ms)
    - Class imbalance (0.1% fraud rate)
    - Adversarial (fraudsters adapt)
    - False positives costly (lost revenue)
    """
    
    def __init__(self):
        self.system_stats = {
            "transactions": "Billions per year",
            "fraud_rate": "0.1-0.5%",
            "latency_requirement": "<100ms P99",
            "false_positive_cost": "$50-100 per declined good transaction",
            "false_negative_cost": "$500-5000 per fraud"
        }
    
    def ml_approach (self):
        """
        ML approach to fraud detection
        """
        print("\\n=== Fraud Detection ML Approach ===\\n")
        
        print("1. Data Imbalance:")
        print("   Problem: 99.5% legitimate, 0.5% fraud")
        print("   Solutions:")
        print("     - SMOTE (oversample fraud)")
        print("     - Class weights (penalize fraud misses)")
        print("     - Focal loss (focus on hard examples)")
        print("     - Anomaly detection models")
        
        print("\\n2. Features:")
        print("   User:")
        print("     - Transaction history")
        print("     - Device fingerprint")
        print("     - Location / IP address")
        print("     - Velocity (transactions per hour)")
        print("   Transaction:")
        print("     - Amount (unusual for user?)")
        print("     - Merchant category")
        print("     - Time of day")
        print("   Network:")
        print("     - Graph features (connected fraud rings)")
        
        print("\\n3. Models:")
        print("   - Gradient Boosting (XGBoost/LightGBM)")
        print("   - Neural Networks (for complex patterns)")
        print("   - Isolation Forest (anomaly detection)")
        print("   - Graph Neural Networks (fraud rings)")
        print("   → Ensemble for final decision")
        
        print("\\n4. Real-Time Scoring:")
        print("   - Model hosted in-memory")
        print("   - Features pre-aggregated in Redis")
        print("   - P95 latency: 50ms")
        print("   - Fallback to rules if model fails")
    
    def handling_adversaries (self):
        """
        Dealing with adaptive adversaries
        """
        print("\\n=== Handling Adaptive Adversaries ===\\n")
        
        print("Problem:")
        print("  - Fraudsters learn and adapt")
        print("  - Steal credentials → blocked")
        print("  - Find new attack vector")
        
        print("\\nSolutions:")
        
        print("\\n1. Continuous Retraining:")
        print("   - Daily retraining on latest fraud patterns")
        print("   - Online learning for fast adaptation")
        print("   - A/B test new models in shadow mode")
        
        print("\\n2. Ensemble Diversity:")
        print("   - Multiple models with different features")
        print("   - Harder for fraudsters to game all models")
        
        print("\\n3. Human-in-the-Loop:")
        print("   - Review team for uncertain cases")
        print("   - Feedback loop to improve model")
        print("   - Manual rule updates for new patterns")
        
        print("\\n4. Graph Analysis:")
        print("   - Detect fraud rings (connected accounts)")
        print("   - Shared devices, IPs, addresses")
        print("   - One fraud → check network")
    
    def cost_optimization (self):
        """
        Optimizing for business metrics
        """
        print("\\n=== Cost-Aware Optimization ===\\n")
        
        print("Standard ML: Minimize misclassifications")
        print("  Loss = FP + FN")
        
        print("\\nBusiness Reality: Different costs")
        print("  Loss = ($100 × FP) + ($1000 × FN)")
        
        print("\\nSolution: Cost-sensitive learning")
        print("  - Penalize fraud misses 10x more")
        print("  - Adjust decision threshold")
        print("  - Optimize for $ saved, not accuracy")
        
        print("\\nThreshold Tuning:")
        print("  - Low threshold (0.3): Catch more fraud, more FPs")
        print("  - High threshold (0.7): Fewer FPs, miss some fraud")
        print("  - Optimal: Depends on cost ratio")


# Run case study
fraud = FraudDetectionSystem()
fraud.ml_approach()
fraud.handling_adversaries()
fraud.cost_optimization()
\`\`\`

---

## Case Study 5: Real-Time Bidding (RTB)

### Ad Tech at Scale

\`\`\`python
"""
Real-Time Bidding System
"""

class RTBSystem:
    """
    Ad auction system (Google, Facebook)
    
    Constraints:
    - Latency: <100ms for entire auction
    - Scale: Millions auctions per second
    - Budget constraints
    - Click-through-rate prediction
    """
    
    def __init__(self):
        self.requirements = {
            "latency_budget": {
                "total": "100ms",
                "breakdown": {
                    "receive_request": "5ms",
                    "feature_lookup": "10ms",
                    "model_inference": "30ms",
                    "bid_calculation": "5ms",
                    "response": "5ms",
                    "buffer": "45ms"
                }
            },
            "scale": {
                "auctions_per_second": "10 million",
                "advertisers": "1 million+",
                "users": "Billions"
            }
        }
    
    def ctr_prediction (self):
        """
        Click-through-rate prediction
        """
        print("\\n=== CTR Prediction Model ===\\n")
        
        print("Task: Predict P(click | ad, user, context)")
        
        print("\\nFeatures:")
        print("  User:")
        print("    - Demographics")
        print("    - Interest")
        print("    - Browsing history")
        print("  Ad:")
        print("    - Creative features")
        print("    - Historical CTR")
        print("    - Advertiser quality")
        print("  Context:")
        print("    - Page content")
        print("    - Time/location")
        print("    - Device type")
        
        print("\\nModel: Logistic Regression → Deep Learning")
        print("  Early: Logistic regression (fast, simple)")
        print("  Now: Deep & Wide networks")
        print("    - Wide: Memorization (feature crosses)")
        print("    - Deep: Generalization (embeddings)")
        
        print("\\nTraining:")
        print("  - Billions of examples per day")
        print("  - Distributed training (100s GPUs)")
        print("  - Hourly retraining (fresh data)")
        
        print("\\nServing:")
        print("  - Model sharding across servers")
        print("  - Feature caching (Redis)")
        print("  - P99 latency: 15ms")
    
    def bidding_strategy (self):
        """
        Optimal bidding
        """
        print("\\n=== Bidding Strategy ===\\n")
        
        print("Auction: Second-price auction")
        print("  - Bid your true value")
        print("  - Pay second-highest bid")
        
        print("\\nBid Calculation:")
        print("  Bid = P(click) × Value_per_click × Bid_modifier")
        print("  Example:")
        print("    P(click) = 0.02 (2% CTR)")
        print("    Value = $1.00")
        print("    Modifier = 1.2 (want more traffic)")
        print("    Bid = 0.02 × $1.00 × 1.2 = $0.024")
        
        print("\\nBudget Pacing:")
        print("  - Daily budget: $1000")
        print("  - Current spend: $400")
        print("  - Time: 2pm (8 hours remaining)")
        print("  - Remaining: $600 / 8 hours")
        print("  → Adjust bids to pace evenly")
    
    def lessons (self):
        """
        RTB lessons
        """
        print("\\n=== Key Lessons ===\\n")
        
        lessons = {
            "Latency Budget": "Every ms counts - must be fast",
            "Feature Engineering": "Cross-features crucial for CTR",
            "Fresh Data": "Hourly retraining for best performance",
            "Calibration": "Predicted CTR must match actual CTR",
            "Budget Pacing": "ML for budgets, not just CTR",
            "Scale": "Distributed everything (training, serving)"
        }
        
        for key, value in lessons.items():
            print(f"✓ {key}: {value}")


# Run case study
rtb = RTBSystem()
rtb.ctr_prediction()
rtb.bidding_strategy()
rtb.lessons()
\`\`\`

---

## Common Patterns Across Systems

### Universal Lessons

\`\`\`python
"""
Patterns that appear across all systems
"""

common_patterns = {
    "Architecture": {
        "pattern": "Multi-layer caching",
        "examples": [
            "Netflix: CDN → Redis → Database",
            "Uber: Memory → Redis → Hive",
            "Trading: Redis → TimescaleDB"
        ],
        "lesson": "Cache hot data at multiple levels"
    },
    
    "Models": {
        "pattern": "Ensemble over single model",
        "examples": [
            "Netflix: CF + Content + Popularity",
            "Fraud: XGBoost + NN + Isolation Forest",
            "RTB: Deep & Wide"
        ],
        "lesson": "Multiple models > single best model"
    },
    
    "Training": {
        "pattern": "Frequent retraining",
        "examples": [
            "Uber: Daily retraining",
            "Fraud: Daily/hourly",
            "RTB: Hourly"
        ],
        "lesson": "Fresh data = better models"
    },
    
    "Deployment": {
        "pattern": "Canary/gradual rollout",
        "examples": [
            "Netflix: 1000s A/B tests",
            "Uber: Canary deployments",
            "Fraud: Shadow mode first"
        ],
        "lesson": "Never deploy to 100% immediately"
    },
    
    "Monitoring": {
        "pattern": "Business metrics > ML metrics",
        "examples": [
            "Netflix: Engagement, not just accuracy",
            "Fraud: $ saved, not F1-score",
            "Trading: Sharpe ratio, not RMSE"
        ],
        "lesson": "Optimize for business outcomes"
    }
}

print("\\n=== Universal ML System Patterns ===\\n")

for category, details in common_patterns.items():
    print(f"{category}: {details['pattern']}")
    print(f"  Examples:")
    for ex in details['examples']:
        print(f"    - {ex}")
    print(f"  → {details['lesson']}")
    print()
\`\`\`

---

## Key Takeaways

1. **Netflix**: Ensemble of algorithms, massive A/B testing culture
2. **Uber**: Feature store is critical infrastructure, not just models
3. **Trading**: Latency is everything, simple models win
4. **Fraud**: Adversarial environment, continuous adaptation needed
5. **RTB**: Scale requires distribution at every layer

**Common Themes**:
- ✅ Multi-level caching for performance
- ✅ Ensemble > single model
- ✅ Frequent retraining with fresh data
- ✅ Gradual deployment with A/B testing
- ✅ Monitor business metrics, not just ML metrics
- ✅ Infrastructure matters as much as algorithms

**For Your System**:
- Start simple, add complexity only when needed
- Build for observability from day one
- Test everything before production
- Plan for failure (fallbacks, circuit breakers)
- Optimize for business impact, not accuracy

**Next Steps**: With real-world examples covered, we'll explore MLOps best practices.
`,
};
