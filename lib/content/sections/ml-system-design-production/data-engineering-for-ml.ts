export const dataEngineeringForML = {
  title: 'Data Engineering for ML',
  id: 'data-engineering-for-ml',
  content: `
# Data Engineering for ML

## Introduction

**"Data is the new oil, but like oil, it needs to be refined."**

In production ML systems, **data engineering often takes 60-80% of the effort**. Poor data engineering leads to:
- **Garbage in, garbage out**: Best model can't overcome bad data
- **Training-serving skew**: Model works in training, fails in production
- **Data quality issues**: Missing values, duplicates, outliers
- **Scalability problems**: Can't handle growing data volumes

This section covers building **robust, scalable data pipelines** for machine learning, with emphasis on trading systems where data quality and freshness are critical.

### The Data Pipeline

\`\`\`
Raw Data → Ingestion → Validation → Transform → Feature Store → Model
     ↓          ↓            ↓            ↓            ↓            ↓
 Sources    Quality      Schema      Features     Cache      Training
           Checks       Validation  Engineering   Redis      /Serving
\`\`\`

By the end of this section, you'll understand:
- Designing scalable data pipelines
- Data versioning and lineage
- Feature stores and caching strategies
- ETL vs ELT patterns
- Data quality monitoring
- Real-time vs batch data processing

---

## Data Pipeline Architecture

### Modern Data Stack for ML

\`\`\`python
"""
Complete Data Pipeline for Trading System
"""

class MLDataPipeline:
    """
    End-to-end data pipeline for ML trading system
    
    Components:
    1. Data ingestion (batch + streaming)
    2. Data validation and quality checks
    3. Feature engineering
    4. Feature store
    5. Model training data
    6. Serving data (real-time)
    """
    
    def __init__(self):
        self.ingestion_layer = DataIngestion()
        self.validation_layer = DataValidation()
        self.transformation_layer = FeatureEngineering()
        self.storage_layer = FeatureStore()
        self.monitoring = DataQualityMonitor()
    
    def batch_pipeline (self, date):
        """
        Daily batch pipeline
        Runs at 6am daily, processes previous day's data
        """
        print(f"\\n=== Batch Pipeline: {date} ===")
        
        # 1. Ingest raw data
        raw_data = self.ingestion_layer.ingest_batch (date)
        print(f"✓ Ingested {len (raw_data)} records")
        
        # 2. Validate
        validation_result = self.validation_layer.validate (raw_data)
        if not validation_result['valid']:
            raise ValueError (f"Data validation failed: {validation_result['errors']}")
        print(f"✓ Validation passed")
        
        # 3. Transform
        features = self.transformation_layer.engineer_features (raw_data)
        print(f"✓ Engineered {len (features.columns)} features")
        
        # 4. Store
        self.storage_layer.write_batch (features, date)
        print(f"✓ Stored to feature store")
        
        # 5. Monitor
        metrics = self.monitoring.compute_metrics (features)
        self.monitoring.alert_if_drift (metrics)
        print(f"✓ Quality metrics logged")
        
        return features
    
    def streaming_pipeline (self, event):
        """
        Real-time streaming pipeline
        Processes incoming market data events
        """
        # 1. Validate event
        if not self.validation_layer.validate_event (event):
            return None
        
        # 2. Transform (fast features only)
        features = self.transformation_layer.engineer_features_realtime (event)
        
        # 3. Update cache
        self.storage_layer.update_cache (features)
        
        return features


class DataIngestion:
    """
    Data ingestion layer
    """
    
    def ingest_batch (self, date):
        """
        Batch ingestion from multiple sources
        """
        import pandas as pd
        
        # Source 1: Market data (S3, database)
        market_data = self._load_market_data (date)
        
        # Source 2: Alternative data (APIs)
        sentiment_data = self._load_sentiment_data (date)
        
        # Source 3: News data
        news_data = self._load_news_data (date)
        
        # Merge all sources
        data = pd.merge (market_data, sentiment_data, on='symbol', how='left')
        data = pd.merge (data, news_data, on='symbol', how='left')
        
        return data
    
    def _load_market_data (self, date):
        """Load from database or S3"""
        import pandas as pd
        # Simulated
        return pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'open': [150, 2800, 300],
            'high': [152, 2850, 305],
            'low': [149, 2790, 298],
            'close': [151, 2820, 303],
            'volume': [1e7, 5e6, 8e6],
            'date': [date] * 3
        })
    
    def _load_sentiment_data (self, date):
        """Load from sentiment API"""
        import pandas as pd
        return pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'sentiment': [0.65, 0.72, 0.58]
        })
    
    def _load_news_data (self, date):
        """Load from news API"""
        import pandas as pd
        return pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'news_count': [15, 8, 12]
        })


# Example usage
pipeline = MLDataPipeline()
# features = pipeline.batch_pipeline('2024-01-15')
\`\`\`

---

## Data Validation and Quality

### Schema Validation

\`\`\`python
"""
Data Schema and Validation
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

class DataValidator:
    """
    Comprehensive data validation
    """
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
    
    def validate (self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all validation checks
        """
        errors = []
        warnings = []
        
        # 1. Schema validation
        schema_errors = self._validate_schema (data)
        errors.extend (schema_errors)
        
        # 2. Data quality checks
        quality_issues = self._check_data_quality (data)
        warnings.extend (quality_issues)
        
        # 3. Business logic validation
        business_errors = self._validate_business_rules (data)
        errors.extend (business_errors)
        
        # 4. Statistical validation
        stat_warnings = self._check_statistical_properties (data)
        warnings.extend (stat_warnings)
        
        return {
            'valid': len (errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'summary': {
                'total_records': len (data),
                'error_count': len (errors),
                'warning_count': len (warnings)
            }
        }
    
    def _validate_schema (self, data: pd.DataFrame) -> List[str]:
        """
        Check schema compliance
        """
        errors = []
        
        # Check required columns
        required_cols = self.schema.get('required_columns', [])
        missing_cols = set (required_cols) - set (data.columns)
        if missing_cols:
            errors.append (f"Missing columns: {missing_cols}")
        
        # Check data types
        for col, expected_dtype in self.schema.get('dtypes', {}).items():
            if col in data.columns:
                actual_dtype = str (data[col].dtype)
                if expected_dtype not in actual_dtype:
                    errors.append(
                        f"Column '{col}': expected {expected_dtype}, "
                        f"got {actual_dtype}"
                    )
        
        return errors
    
    def _check_data_quality (self, data: pd.DataFrame) -> List[str]:
        """
        Data quality checks
        """
        warnings = []
        
        # 1. Missing values
        missing_pct = data.isnull().sum() / len (data) * 100
        high_missing = missing_pct[missing_pct > 5]
        if not high_missing.empty:
            warnings.append(
                f"High missing rate: {high_missing.to_dict()}"
            )
        
        # 2. Duplicates
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            warnings.append (f"Found {duplicates} duplicate records")
        
        # 3. Outliers (for numeric columns)
        numeric_cols = data.select_dtypes (include=[np.number]).columns
        for col in numeric_cols:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            outliers = (z_scores > 5).sum()
            if outliers > 0:
                warnings.append(
                    f"Column '{col}': {outliers} outliers (z-score > 5)"
                )
        
        return warnings
    
    def _validate_business_rules (self, data: pd.DataFrame) -> List[str]:
        """
        Business logic validation
        """
        errors = []
        
        # Example: Price checks
        if 'close' in data.columns:
            # Prices must be positive
            negative_prices = (data['close'] <= 0).sum()
            if negative_prices > 0:
                errors.append (f"Found {negative_prices} non-positive prices")
            
            # Price change sanity check (< 50% per day)
            if 'open' in data.columns:
                price_change_pct = abs(
                    (data['close'] - data['open']) / data['open'] * 100
                )
                extreme_changes = (price_change_pct > 50).sum()
                if extreme_changes > 0:
                    errors.append(
                        f"Found {extreme_changes} extreme price changes (>50%)"
                    )
        
        # Volume checks
        if 'volume' in data.columns:
            zero_volume = (data['volume'] == 0).sum()
            if zero_volume > 0:
                errors.append (f"Found {zero_volume} records with zero volume")
        
        return errors
    
    def _check_statistical_properties (self, data: pd.DataFrame) -> List[str]:
        """
        Statistical property checks
        """
        warnings = []
        
        # Check for constant columns
        numeric_cols = data.select_dtypes (include=[np.number]).columns
        for col in numeric_cols:
            if data[col].std() == 0:
                warnings.append (f"Column '{col}' has zero variance")
        
        # Check for high correlation (potential redundancy)
        if len (numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr().abs()
            # Set diagonal to 0
            np.fill_diagonal (corr_matrix.values, 0)
            
            high_corr = corr_matrix > 0.95
            if high_corr.any().any():
                pairs = []
                for i in range (len (corr_matrix)):
                    for j in range (i+1, len (corr_matrix)):
                        if corr_matrix.iloc[i, j] > 0.95:
                            pairs.append(
                                f"({corr_matrix.index[i]}, "
                                f"{corr_matrix.columns[j]})"
                            )
                if pairs:
                    warnings.append (f"High correlation pairs: {pairs[:3]}")
        
        return warnings


# Define schema
trading_schema = {
    'required_columns': ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'],
    'dtypes': {
        'symbol': 'object',
        'date': 'datetime',
        'open': 'float',
        'high': 'float',
        'low': 'float',
        'close': 'float',
        'volume': 'int'
    }
}

# Create validator
validator = DataValidator (trading_schema)

# Test data
test_data = pd.DataFrame({
    'symbol': ['AAPL', 'GOOGL', 'MSFT'],
    'date': pd.to_datetime(['2024-01-15'] * 3),
    'open': [150.0, 2800.0, 300.0],
    'high': [152.0, 2850.0, 305.0],
    'low': [149.0, 2790.0, 298.0],
    'close': [151.0, 2820.0, 303.0],
    'volume': [10000000, 5000000, 8000000]
})

# Validate
result = validator.validate (test_data)
print(f"\\nValidation result: {result['valid']}")
print(f"Errors: {len (result['errors'])}")
print(f"Warnings: {len (result['warnings'])}")

if result['errors']:
    print("\\nErrors:")
    for error in result['errors']:
        print(f"  - {error}")

if result['warnings']:
    print("\\nWarnings:")
    for warning in result['warnings'][:5]:  # Show first 5
        print(f"  - {warning}")
\`\`\`

---

## Data Versioning and Lineage

### Data Version Control with DVC

\`\`\`python
"""
Data Versioning System
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, Any
import pandas as pd

class DataVersionControl:
    """
    Track data versions for reproducibility
    
    Inspired by DVC (Data Version Control)
    """
    
    def __init__(self, storage_path='./data_versions'):
        self.storage_path = storage_path
        self.versions = {}
    
    def create_version(
        self,
        data: pd.DataFrame,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Create new data version
        
        Returns version ID
        """
        # Generate version ID
        version_id = self._generate_version_id (data, metadata)
        
        # Create version record
        version_record = {
            'version_id': version_id,
            'timestamp': datetime.now().isoformat(),
            'shape': data.shape,
            'columns': list (data.columns),
            'dtypes': {col: str (dtype) for col, dtype in data.dtypes.items()},
            'hash': self._compute_hash (data),
            'metadata': metadata,
            'stats': self._compute_stats (data)
        }
        
        # Store version
        self.versions[version_id] = version_record
        
        # In production: save data to S3/blob storage
        # self._save_to_storage (data, version_id)
        
        print(f"✓ Created data version: {version_id}")
        
        return version_id
    
    def _generate_version_id (self, data: pd.DataFrame, metadata: Dict) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_hash = self._compute_hash (data)[:8]
        return f"v_{timestamp}_{data_hash}"
    
    def _compute_hash (self, data: pd.DataFrame) -> str:
        """Compute data fingerprint"""
        # Hash based on data content
        data_bytes = pd.util.hash_pandas_object (data).values.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()
    
    def _compute_stats (self, data: pd.DataFrame) -> Dict:
        """Compute summary statistics"""
        numeric_cols = data.select_dtypes (include=[np.number]).columns
        
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': float (data[col].mean()),
                'std': float (data[col].std()),
                'min': float (data[col].min()),
                'max': float (data[col].max()),
                'missing_pct': float (data[col].isnull().sum() / len (data) * 100)
            }
        
        return stats
    
    def get_version_info (self, version_id: str) -> Dict:
        """Get information about a version"""
        return self.versions.get (version_id)
    
    def compare_versions (self, v1: str, v2: str) -> Dict:
        """
        Compare two data versions
        """
        info1 = self.get_version_info (v1)
        info2 = self.get_version_info (v2)
        
        if not info1 or not info2:
            return {"error": "Version not found"}
        
        comparison = {
            'shape_change': {
                'v1': info1['shape'],
                'v2': info2['shape'],
                'rows_diff': info2['shape'][0] - info1['shape'][0],
                'cols_diff': info2['shape'][1] - info1['shape'][1]
            },
            'columns_added': set (info2['columns']) - set (info1['columns']),
            'columns_removed': set (info1['columns']) - set (info2['columns']),
            'stats_comparison': self._compare_stats (info1['stats'], info2['stats'])
        }
        
        return comparison
    
    def _compare_stats (self, stats1: Dict, stats2: Dict) -> Dict:
        """Compare statistics between versions"""
        comparison = {}
        
        common_cols = set (stats1.keys()) & set (stats2.keys())
        
        for col in common_cols:
            s1 = stats1[col]
            s2 = stats2[col]
            
            comparison[col] = {
                'mean_change': s2['mean'] - s1['mean'],
                'std_change': s2['std'] - s1['std'],
                'mean_pct_change': (s2['mean'] - s1['mean']) / s1['mean'] * 100 if s1['mean'] != 0 else 0
            }
        
        return comparison


# Example usage
dvc = DataVersionControl()

# Version 1: Initial data
data_v1 = pd.DataFrame({
    'symbol': ['AAPL', 'GOOGL', 'MSFT'],
    'price': [150.0, 2800.0, 300.0],
    'volume': [1e7, 5e6, 8e6]
})

v1 = dvc.create_version(
    data_v1,
    metadata={
        'source': 'yahoo_finance',
        'date': '2024-01-15',
        'transformations': ['remove_outliers']
    }
)

# Version 2: Updated data
data_v2 = pd.DataFrame({
    'symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
    'price': [151.0, 2820.0, 303.0, 180.0],
    'volume': [1.1e7, 5.2e6, 8.5e6, 2e7],
    'sentiment': [0.65, 0.72, 0.58, 0.80]  # New column
})

v2 = dvc.create_version(
    data_v2,
    metadata={
        'source': 'yahoo_finance',
        'date': '2024-01-16',
        'transformations': ['remove_outliers', 'add_sentiment']
    }
)

# Compare versions
comparison = dvc.compare_versions (v1, v2)
print(f"\\n=== Version Comparison ===")
print(f"Shape change: {comparison['shape_change']}")
print(f"Columns added: {comparison['columns_added']}")
print(f"Columns removed: {comparison['columns_removed']}")
\`\`\`

---

## Feature Store

### Building a Feature Store

\`\`\`python
"""
Feature Store Implementation
"""

import redis
import pickle
from typing import Dict, List, Any
import pandas as pd
import time

class FeatureStore:
    """
    Feature store for ML systems
    
    Purposes:
    1. Centralized feature storage
    2. Consistent features for training and serving
    3. Fast feature access (Redis cache)
    4. Feature versioning
    5. Feature sharing across models
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        # In production: connect to Redis
        # self.redis_client = redis.Redis (host=redis_host, port=redis_port)
        
        # For demo: in-memory dict
        self.redis_client = {}
        self.offline_store = {}  # Simulates S3/database
    
    def write_batch_features(
        self,
        entity: str,
        features: pd.DataFrame,
        feature_group: str,
        timestamp_col: str = 'timestamp'
    ):
        """
        Write features to offline store (for training)
        
        Args:
            entity: Entity type (e.g., 'symbol', 'user_id')
            features: Feature DataFrame
            feature_group: Group name (e.g., 'technical_indicators')
            timestamp_col: Timestamp column name
        """
        # Store in offline store (S3, database)
        key = f"offline:{entity}:{feature_group}"
        self.offline_store[key] = features.copy()
        
        print(f"✓ Wrote {len (features)} records to offline store")
        print(f"  Entity: {entity}, Group: {feature_group}")
    
    def write_online_features(
        self,
        entity_id: str,
        features: Dict[str, Any],
        feature_group: str,
        ttl: int = 3600
    ):
        """
        Write features to online store (for serving)
        
        Args:
            entity_id: Specific entity (e.g., 'AAPL')
            features: Feature dict
            feature_group: Group name
            ttl: Time-to-live in seconds
        """
        # Store in Redis for fast access
        key = f"online:{entity_id}:{feature_group}"
        
        # Serialize features
        value = pickle.dumps({
            'features': features,
            'timestamp': time.time(),
            'ttl': ttl
        })
        
        # In production: self.redis_client.setex (key, ttl, value)
        self.redis_client[key] = value
        
        print(f"✓ Cached features for {entity_id} (TTL: {ttl}s)")
    
    def get_online_features(
        self,
        entity_id: str,
        feature_group: str,
        feature_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get features from online store (real-time serving)
        
        Returns features within milliseconds
        """
        key = f"online:{entity_id}:{feature_group}"
        
        # Get from Redis
        # In production: value = self.redis_client.get (key)
        value = self.redis_client.get (key)
        
        if value is None:
            # Cache miss: fallback to offline store or recompute
            print(f"⚠️  Cache miss for {entity_id}")
            return {}
        
        # Deserialize
        data = pickle.loads (value)
        features = data['features']
        
        # Filter specific features if requested
        if feature_names:
            features = {k: v for k, v in features.items() if k in feature_names}
        
        return features
    
    def get_training_features(
        self,
        entity: str,
        feature_groups: List[str],
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Get features for training (batch)
        
        Reads from offline store (S3, database)
        """
        all_features = []
        
        for feature_group in feature_groups:
            key = f"offline:{entity}:{feature_group}"
            
            if key in self.offline_store:
                df = self.offline_store[key].copy()
                
                # Filter by date if specified
                if 'timestamp' in df.columns and start_date:
                    df = df[df['timestamp'] >= start_date]
                if 'timestamp' in df.columns and end_date:
                    df = df[df['timestamp'] <= end_date]
                
                all_features.append (df)
        
        if not all_features:
            return pd.DataFrame()
        
        # Merge all feature groups
        result = all_features[0]
        for df in all_features[1:]:
            result = pd.merge (result, df, on=['symbol', 'timestamp'], how='inner')
        
        print(f"✓ Retrieved {len (result)} training samples")
        
        return result
    
    def compute_feature_stats (self, feature_group: str) -> Dict:
        """
        Compute feature statistics for monitoring
        """
        key = f"offline:symbol:{feature_group}"
        
        if key not in self.offline_store:
            return {}
        
        df = self.offline_store[key]
        numeric_cols = df.select_dtypes (include=[np.number]).columns
        
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': float (df[col].mean()),
                'std': float (df[col].std()),
                'min': float (df[col].min()),
                'max': float (df[col].max()),
                'missing_pct': float (df[col].isnull().sum() / len (df) * 100)
            }
        
        return stats


# Example usage
fs = FeatureStore()

# 1. Batch write (daily job)
batch_features = pd.DataFrame({
    'symbol': ['AAPL', 'GOOGL', 'MSFT'],
    'timestamp': pd.to_datetime(['2024-01-15'] * 3),
    'sma_20': [150.5, 2805.2, 301.3],
    'rsi': [65.3, 58.2, 72.1],
    'macd': [2.3, -1.5, 3.1]
})

fs.write_batch_features(
    entity='symbol',
    features=batch_features,
    feature_group='technical_indicators'
)

# 2. Online write (real-time)
fs.write_online_features(
    entity_id='AAPL',
    features={
        'sma_20': 150.5,
        'rsi': 65.3,
        'macd': 2.3,
        'current_price': 151.0
    },
    feature_group='technical_indicators',
    ttl=60  # 1 minute
)

# 3. Online read (serving)
features = fs.get_online_features(
    entity_id='AAPL',
    feature_group='technical_indicators',
    feature_names=['sma_20', 'rsi']
)

print(f"\\nFeatures for AAPL: {features}")

# 4. Training data retrieval
training_data = fs.get_training_features(
    entity='symbol',
    feature_groups=['technical_indicators']
)

print(f"\\nTraining data shape: {training_data.shape}")
\`\`\`

---

## ETL vs ELT Patterns

### ETL (Extract-Transform-Load)

\`\`\`python
"""
ETL Pattern: Transform before loading
"""

class ETLPipeline:
    """
    Traditional ETL: Transform data before storing
    
    Use when:
    - Data needs heavy cleaning
    - Storage is expensive
    - Users need clean data immediately
    """
    
    def extract (self):
        """Extract from sources"""
        # Pull from APIs, databases, files
        raw_data = self._fetch_raw_data()
        return raw_data
    
    def transform (self, raw_data):
        """Transform data (heavy computation)"""
        import pandas as pd
        
        # Clean
        data = self._clean_data (raw_data)
        
        # Validate
        data = self._validate_data (data)
        
        # Engineer features
        data = self._engineer_features (data)
        
        # Aggregate
        data = self._aggregate_data (data)
        
        return data
    
    def load (self, transformed_data):
        """Load clean data to warehouse"""
        # Store in database/data warehouse
        self._save_to_warehouse (transformed_data)
        
        print(f"✓ Loaded {len (transformed_data)} clean records")
    
    def _fetch_raw_data (self):
        """Fetch from sources"""
        import pandas as pd
        return pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL'],
            'price': [150.0, 2800.0],
            'volume': [1e7, 5e6],
            'bad_column': [', ']  # Will be cleaned
        })
    
    def _clean_data (self, data):
        """Data cleaning"""
        # Remove bad columns
        if 'bad_column' in data.columns:
            data = data.drop('bad_column', axis=1)
        
        # Handle missing
        data = data.fillna (method='ffill')
        
        return data
    
    def _validate_data (self, data):
        """Validation"""
        # Remove invalid rows
        data = data[data['price'] > 0]
        data = data[data['volume'] > 0]
        return data
    
    def _engineer_features (self, data):
        """Feature engineering"""
        data['price_volume_ratio'] = data['price'] / data['volume']
        return data
    
    def _aggregate_data (self, data):
        """Aggregation"""
        # Example: daily aggregates
        return data
    
    def _save_to_warehouse (self, data):
        """Save to warehouse"""
        # In practice: write to PostgreSQL, Snowflake, etc.
        pass


# ELT Pattern: Load raw, transform later
class ELTPipeline:
    """
    Modern ELT: Load raw data, transform in warehouse
    
    Use when:
    - Storage is cheap (cloud)
    - Need data lake for multiple use cases
    - Want to keep raw data
    """
    
    def extract (self):
        """Extract from sources"""
        raw_data = self._fetch_raw_data()
        return raw_data
    
    def load (self, raw_data):
        """Load raw data directly to data lake"""
        # Store raw data in S3/data lake
        self._save_to_data_lake (raw_data)
        
        print(f"✓ Loaded {len (raw_data)} raw records to data lake")
    
    def transform (self, query):
        """
        Transform in warehouse using SQL
        
        Advantages:
        - Use warehouse compute power
        - SQL is familiar to analysts
        - Can create multiple views
        """
        # Run transformation in warehouse (BigQuery, Snowflake)
        sql = f"""
        SELECT
            symbol,
            price,
            volume,
            price / volume as price_volume_ratio,
            CASE WHEN price > 100 THEN 'expensive' ELSE 'cheap' END as price_category
        FROM raw_data
        WHERE price > 0 AND volume > 0
        """
        
        # Execute in warehouse
        # transformed = warehouse.query (sql)
        
        print("✓ Transformed data in warehouse")
    
    def _fetch_raw_data (self):
        """Fetch from sources"""
        import pandas as pd
        return pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'price': [150.0, 2800.0, -1.0],  # Keep bad data
            'volume': [1e7, 5e6, 0],  # Keep bad data
            'extra_field': ['a', 'b', 'c']  # Keep everything
        })
    
    def _save_to_data_lake (self, data):
        """Save raw to data lake"""
        # In practice: write to S3 Parquet
        pass


print("=== ETL vs ELT ===")
print("\\nETL:")
etl = ETLPipeline()
raw = etl.extract()
transformed = etl.transform (raw)
etl.load (transformed)

print("\\nELT:")
elt = ELTPipeline()
raw = elt.extract()
elt.load (raw)
elt.transform("transform_query")
\`\`\`

---

## Data Quality Monitoring

\`\`\`python
"""
Data Quality Monitoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

class DataQualityMonitor:
    """
    Monitor data quality over time
    """
    
    def __init__(self):
        self.baseline_stats = {}
        self.alert_thresholds = {
            'missing_rate': 0.10,  # Alert if >10% missing
            'std_change': 0.50,    # Alert if std changes >50%
            'mean_change': 0.30,   # Alert if mean changes >30%
            'volume_drop': 0.50    # Alert if volume drops >50%
        }
    
    def compute_metrics (self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute data quality metrics
        """
        metrics = {
            'timestamp': pd.Timestamp.now(),
            'record_count': len (data),
            'column_count': len (data.columns),
            'missing_values': {},
            'numeric_stats': {},
            'categorical_stats': {}
        }
        
        # Missing values
        for col in data.columns:
            missing_pct = data[col].isnull().sum() / len (data) * 100
            metrics['missing_values'][col] = missing_pct
        
        # Numeric columns
        numeric_cols = data.select_dtypes (include=[np.number]).columns
        for col in numeric_cols:
            metrics['numeric_stats'][col] = {
                'mean': float (data[col].mean()),
                'std': float (data[col].std()),
                'min': float (data[col].min()),
                'max': float (data[col].max()),
                'q25': float (data[col].quantile(0.25)),
                'q50': float (data[col].quantile(0.50)),
                'q75': float (data[col].quantile(0.75))
            }
        
        # Categorical columns
        categorical_cols = data.select_dtypes (include=['object', 'category']).columns
        for col in categorical_cols:
            metrics['categorical_stats'][col] = {
                'unique_count': int (data[col].nunique()),
                'most_common': str (data[col].mode()[0]) if not data[col].mode().empty else None
            }
        
        return metrics
    
    def set_baseline (self, data: pd.DataFrame):
        """
        Set baseline statistics
        """
        self.baseline_stats = self.compute_metrics (data)
        print(f"✓ Baseline set with {len (data)} records")
    
    def detect_drift(
        self,
        current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect data drift compared to baseline
        """
        if not self.baseline_stats:
            return {"error": "No baseline set"}
        
        current_metrics = self.compute_metrics (current_data)
        
        drift_report = {
            'has_drift': False,
            'alerts': [],
            'changes': {}
        }
        
        # Check record count change
        baseline_count = self.baseline_stats['record_count']
        current_count = current_metrics['record_count']
        count_change = (current_count - baseline_count) / baseline_count
        
        if abs (count_change) > self.alert_thresholds['volume_drop']:
            drift_report['alerts'].append(
                f"Record count changed by {count_change*100:.1f}%"
            )
            drift_report['has_drift'] = True
        
        # Check missing rate changes
        for col in current_metrics['missing_values']:
            baseline_missing = self.baseline_stats['missing_values'].get (col, 0)
            current_missing = current_metrics['missing_values'][col]
            
            if current_missing > self.alert_thresholds['missing_rate'] * 100:
                drift_report['alerts'].append(
                    f"{col}: High missing rate {current_missing:.1f}%"
                )
                drift_report['has_drift'] = True
        
        # Check numeric stats changes
        for col in current_metrics['numeric_stats']:
            if col not in self.baseline_stats['numeric_stats']:
                continue
            
            baseline_stats = self.baseline_stats['numeric_stats'][col]
            current_stats = current_metrics['numeric_stats'][col]
            
            # Mean change
            if baseline_stats['mean'] != 0:
                mean_change = abs(
                    (current_stats['mean'] - baseline_stats['mean']) /
                    baseline_stats['mean']
                )
                if mean_change > self.alert_thresholds['mean_change']:
                    drift_report['alerts'].append(
                        f"{col}: Mean changed by {mean_change*100:.1f}%"
                    )
                    drift_report['has_drift'] = True
            
            # Std change
            if baseline_stats['std'] != 0:
                std_change = abs(
                    (current_stats['std'] - baseline_stats['std']) /
                    baseline_stats['std']
                )
                if std_change > self.alert_thresholds['std_change']:
                    drift_report['alerts'].append(
                        f"{col}: Std changed by {std_change*100:.1f}%"
                    )
                    drift_report['has_drift'] = True
            
            # Store changes
            drift_report['changes'][col] = {
                'mean_change_pct': mean_change * 100 if baseline_stats['mean'] != 0 else 0,
                'std_change_pct': std_change * 100 if baseline_stats['std'] != 0 else 0
            }
        
        return drift_report
    
    def alert_if_drift (self, metrics: Dict):
        """
        Send alerts if drift detected
        """
        # In production: send to Slack, PagerDuty, email
        if isinstance (metrics, dict) and metrics.get('has_drift'):
            print("\\n🚨 DATA DRIFT ALERT")
            for alert in metrics['alerts']:
                print(f"  - {alert}")


# Example usage
monitor = DataQualityMonitor()

# Baseline data (Week 1)
baseline_data = pd.DataFrame({
    'symbol': ['AAPL'] * 100,
    'price': np.random.normal(150, 10, 100),
    'volume': np.random.normal(1e7, 1e6, 100),
    'sentiment': np.random.uniform(-1, 1, 100)
})

monitor.set_baseline (baseline_data)

# Current data (Week 2) - with drift
current_data = pd.DataFrame({
    'symbol': ['AAPL'] * 100,
    'price': np.random.normal(170, 15, 100),  # Mean changed
    'volume': np.random.normal(5e6, 1e6, 100),  # Volume dropped
    'sentiment': np.concatenate([
        np.random.uniform(-1, 1, 80),
        [np.nan] * 20  # 20% missing (drift!)
    ])
})

# Detect drift
drift_report = monitor.detect_drift (current_data)

print(f"\\nDrift detected: {drift_report['has_drift']}")
if drift_report['has_drift']:
    print("\\nAlerts:")
    for alert in drift_report['alerts']:
        print(f"  ⚠️  {alert}")
\`\`\`

---

## Key Takeaways

1. **Data Engineering is 60-80% of ML Work**: More time than model training
2. **Data Quality > Model Quality**: Best model can't fix bad data
3. **Validation is Critical**: Schema, business rules, statistical properties
4. **Version Your Data**: Reproducibility requires data versioning (DVC)
5. **Feature Store Pattern**: Centralized features for training/serving consistency
6. **ETL vs ELT**:
   - **ETL**: Clean data before storing (traditional)
   - **ELT**: Store raw, transform later (modern, cloud-native)
7. **Monitor Data Quality**: Detect drift, missing values, distribution shifts
8. **Separate Batch and Real-time**: Different patterns for different latencies

**Trading-Specific Considerations**:
- **Data Freshness**: Critical for alpha decay
- **Point-in-Time Correctness**: No lookahead bias
- **Survivorship Bias**: Include delisted stocks
- **Market Regimes**: Data distribution changes over time

**Next Steps**: With robust data pipelines in place, we'll cover experiment tracking and model training in the next sections.
`,
};
