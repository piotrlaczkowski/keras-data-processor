# 📅 Date Features

<div class="feature-header">
  <div class="feature-title">
    <h2>Date Features in KDP</h2>
    <p>Extract powerful patterns from temporal data like timestamps, dates, and time series.</p>
  </div>
</div>

## 📋 Overview

<div class="overview-card">
  <p>Date features transform timestamps and dates into ML-ready representations that capture important temporal patterns and seasonality. KDP automatically handles date parsing and formatting, enabling your models to learn from time-based signals.</p>
</div>

## 🚀 Date Processing Approaches

<div class="approaches-container">
  <div class="approach-card">
    <span class="approach-icon">📆</span>
    <h3>Component Extraction</h3>
    <p>Breaking dates into day, month, year, etc.</p>
  </div>

  <div class="approach-card">
    <span class="approach-icon">🔄</span>
    <h3>Cyclical Encoding</h3>
    <p>Representing cyclic time components (hour, weekday)</p>
  </div>

  <div class="approach-card">
    <span class="approach-icon">📊</span>
    <h3>Temporal Distances</h3>
    <p>Computing time since reference points</p>
  </div>

  <div class="approach-card">
    <span class="approach-icon">📈</span>
    <h3>Seasonality Analysis</h3>
    <p>Capturing seasonal patterns and trends</p>
  </div>
</div>

## 📝 Basic Usage

<div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType

# Quick date feature definition
features = {
    "purchase_date": FeatureType.DATE,     # Transaction dates
    "signup_date": FeatureType.DATE,       # User signup dates
    "last_active": FeatureType.DATE        # Last activity timestamps
}

# Create your preprocessor
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs=features
)
```

</div>

## 🧠 Advanced Configuration

<div class="advanced-section">
  <p>For more control over date processing, use the <code>DateFeature</code> class:</p>

  <div class="code-container">

```python
from kdp.features import DateFeature

features = {
    # Transaction date with component extraction
    "transaction_date": DateFeature(
        name="transaction_date",
        feature_type=FeatureType.DATE,
        add_day_of_week=True,      # Extract day of week
        add_month=True,            # Extract month
        add_quarter=True,          # Extract quarter
        cyclical_encoding=True     # Use sine/cosine encoding for cyclical features
    ),

    # User signup date with time since reference
    "signup_date": DateFeature(
        name="signup_date",
        feature_type=FeatureType.DATE,
        add_time_since_reference=True,
        reference_date="2020-01-01"  # Reference point
    ),

    # Event timestamp with hour component
    "event_timestamp": DateFeature(
        name="event_timestamp",
        feature_type=FeatureType.DATE,
        add_hour=True,             # Extract hour
        add_day_of_week=True,      # Extract day of week
        add_is_weekend=True        # Add weekend indicator
    )
}
```

  </div>
</div>

## ⚙️ Key Configuration Parameters

<div class="table-container">
  <table class="config-table">
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Description</th>
        <th>Default</th>
        <th>Options</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>add_year</code></td>
        <td>Extract year component</td>
        <td>False</td>
        <td>Boolean</td>
      </tr>
      <tr>
        <td><code>add_month</code></td>
        <td>Extract month component</td>
        <td>False</td>
        <td>Boolean</td>
      </tr>
      <tr>
        <td><code>add_day</code></td>
        <td>Extract day component</td>
        <td>False</td>
        <td>Boolean</td>
      </tr>
      <tr>
        <td><code>add_day_of_week</code></td>
        <td>Extract day of week</td>
        <td>False</td>
        <td>Boolean</td>
      </tr>
      <tr>
        <td><code>add_hour</code></td>
        <td>Extract hour component</td>
        <td>False</td>
        <td>Boolean</td>
      </tr>
      <tr>
        <td><code>cyclical_encoding</code></td>
        <td>Use sine/cosine encoding</td>
        <td>False</td>
        <td>Boolean</td>
      </tr>
      <tr>
        <td><code>add_is_weekend</code></td>
        <td>Add weekend indicator</td>
        <td>False</td>
        <td>Boolean</td>
      </tr>
    </tbody>
  </table>
</div>

## 💡 Powerful Features

<div class="power-features">
  <div class="power-feature-card">
    <h3>🔄 Cyclical Encoding</h3>
    <p>Properly represent cyclical time components (like hour, day of week) using sine/cosine transformations:</p>
    <div class="code-container">

```python
# Configure cyclical encoding for time components
date_feature = DateFeature(
    name="event_time",
    feature_type=FeatureType.DATE,
    add_hour=True,
    add_day_of_week=True,
    cyclical_encoding=True  # Enable cyclical encoding
)

# Create preprocessor with cyclical date features
preprocessor = PreprocessingModel(
    path_data="events.csv",
    features_specs={"event_time": date_feature}
)
```

    </div>
  </div>

  <div class="power-feature-card">
    <h3>📏 Time-Since Features</h3>
    <p>Calculate time since reference points for meaningful temporal distances:</p>
    <div class="code-container">

```python
# Compute days since reference date
date_feature = DateFeature(
    name="signup_date",
    feature_type=FeatureType.DATE,
    add_time_since_reference=True,
    reference_date="2020-01-01",     # Fixed reference
    time_since_unit="days"           # Unit for calculation
)

# Compute time since multiple references
preprocessor = PreprocessingModel(
    path_data="user_data.csv",
    features_specs={
        "signup_date": date_feature,
        "last_purchase": DateFeature(
            name="last_purchase",
            add_time_since_reference=True,
            reference_date="today",  # Dynamic reference (current date)
            time_since_unit="days"
        )
    }
)
```

    </div>
  </div>
</div>

## 🔧 Real-World Examples

<div class="examples-container">
  <div class="example-card">
    <h3>E-commerce Purchase Analysis</h3>
    <div class="code-container">

```python
# Analyze purchase patterns over time
from kdp.features import DateFeature, NumericalFeature, CategoricalFeature

preprocessor = PreprocessingModel(
    path_data="ecommerce_data.csv",
    features_specs={
        # Purchase date with rich time components
        "purchase_date": DateFeature(
            name="purchase_date",
            add_day_of_week=True,
            add_hour=True,
            add_month=True,
            add_is_weekend=True,
            cyclical_encoding=True
        ),

        # User signup date to determine user tenure
        "user_signup_date": DateFeature(
            name="user_signup_date",
            add_time_since_reference=True,
            reference_date="today",
            time_since_unit="days"
        ),

        # Additional features
        "product_category": CategoricalFeature(
            name="product_category",
            feature_type=FeatureType.STRING_CATEGORICAL
        ),
        "purchase_amount": NumericalFeature(
            name="purchase_amount",
            feature_type=FeatureType.FLOAT_RESCALED
        )
    },

    # Define crosses to capture time-based patterns
    feature_crosses=[
        ("purchase_date_day_of_week", "product_category", 16)
    ]
)
```

    </div>
  </div>

  <div class="example-card">
    <h3>Time Series Forecasting</h3>
    <div class="code-container">

```python
# Time series feature extraction for forecasting
preprocessor = PreprocessingModel(
    path_data="sensor_readings.csv",
    features_specs={
        # Timestamp with multiple components
        "timestamp": DateFeature(
            name="timestamp",
            add_year=True,
            add_month=True,
            add_day=True,
            add_hour=True,
            add_day_of_week=True,
            cyclical_encoding=True
        ),

        # Numerical features to predict
        "value": NumericalFeature(
            name="value",
            feature_type=FeatureType.FLOAT_RESCALED,
            use_distribution_aware=True
        ),

        # Additional context features
        "sensor_id": CategoricalFeature(
            name="sensor_id",
            feature_type=FeatureType.STRING_CATEGORICAL
        )
    },

    # Enable tabular attention for discovering temporal patterns
    tabular_attention=True
)
```

    </div>
  </div>
</div>

## 💎 Pro Tips

<div class="pro-tips-grid">
  <div class="pro-tip-card">
    <h3>🔍 Date Format Handling</h3>
    <p>KDP automatically handles common date formats, but you can specify custom formats:</p>
    <div class="code-container">

```python
# Handle custom date formats
from datetime import datetime
import pandas as pd

# Convert dates to standard format before feeding to KDP
def standardize_date(date_str):
    try:
        # Try parsing custom format
        dt = datetime.strptime(date_str, "%d-%b-%Y")
        return dt.strftime("%Y-%m-%d")
    except:
        return date_str

# Apply standardization to your data
data = pd.read_csv("custom_dates.csv")
data["standard_date"] = data["custom_date"].apply(standardize_date)

# Use standardized dates in KDP
preprocessor = PreprocessingModel(
    path_data=data,
    features_specs={"standard_date": FeatureType.DATE}
)
```

    </div>
  </div>

  <div class="pro-tip-card">
    <h3>🧠 Feature Selection</h3>
    <p>Use feature selection to identify important temporal patterns:</p>
    <div class="code-container">

```python
# Determine which date components matter most
preprocessor = PreprocessingModel(
    path_data="events.csv",
    features_specs={
        "event_date": DateFeature(
            name="event_date",
            # Extract all potentially relevant components
            add_year=True,
            add_quarter=True,
            add_month=True,
            add_day=True,
            add_day_of_week=True,
            add_hour=True,
            add_is_weekend=True
        )
    },
    # Enable feature selection to identify important components
    use_feature_selection=True,
    feature_selection_strategy="gradient_based"
)

# After training, check feature importance
result = preprocessor.build_preprocessor()
importance = result["feature_importance"]
print("Most important date components:", importance)
```

    </div>
  </div>

  <div class="pro-tip-card">
    <h3>➕ Cross Features</h3>
    <p>Create crosses with date components to capture context-dependent patterns:</p>
    <div class="code-container">

```python
# Cross date components with categorical features
preprocessor = PreprocessingModel(
    path_data="transactions.csv",
    features_specs={
        # Date with components
        "transaction_date": DateFeature(
            name="transaction_date",
            add_day_of_week=True,
            add_hour=True,
            add_is_weekend=True
        ),

        # Categorical context
        "store_location": FeatureType.STRING_CATEGORICAL,
        "product_category": FeatureType.STRING_CATEGORICAL
    },

    # Define crosses to capture contextual patterns
    feature_crosses=[
        # Weekend shopping differs by location
        ("transaction_date_is_weekend", "store_location", 16),

        # Day of week impacts product category popularity
        ("transaction_date_day_of_week", "product_category", 32),

        # Hour of day impacts product selections
        ("transaction_date_hour", "product_category", 32)
    ]
)
```

    </div>
  </div>

  <div class="pro-tip-card">
    <h3>🌍 Handling Timezones</h3>
    <p>Standardize timezone handling for consistent date processing:</p>
    <div class="code-container">

```python
# Standardize timezones before processing
import pandas as pd
from datetime import datetime
import pytz

# Convert timestamps to a standard timezone
def standardize_timezone(timestamp_str, from_tz='UTC', to_tz='America/New_York'):
    if pd.isna(timestamp_str):
        return None

    # Parse timestamp and set timezone
    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = pytz.timezone(from_tz).localize(dt)

    # Convert to target timezone
    dt = dt.astimezone(pytz.timezone(to_tz))
    return dt.isoformat()

# Apply timezone standardization
data = pd.read_csv("global_events.csv")
data["standardized_time"] = data["event_timestamp"].apply(
    lambda x: standardize_timezone(x, from_tz='UTC', to_tz='America/New_York')
)

# Use standardized timestamps in KDP
preprocessor = PreprocessingModel(
    path_data=data,
    features_specs={"standardized_time": FeatureType.DATE}
)
```

    </div>
  </div>
</div>

## 📊 Model Architecture

<div class="architecture-diagram">
  <div class="mermaid">
    graph TD
      A[Raw Date Data] -->|Parsing| B[Date Components]
      B -->|Cyclical Encoding| C1[Sine/Cosine Components]
      B -->|Direct Encoding| C2[Normalized Components]
      B -->|Reference Distance| C3[Time Since Features]

      C1 --> D[Date Representation]
      C2 --> D
      C3 --> D

      style A fill:#f9f9f9,stroke:#ccc,stroke-width:2px
      style B fill:#e3f2fd,stroke:#64b5f6,stroke-width:2px
      style C1 fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px
      style C2 fill:#fff8e1,stroke:#ffd54f,stroke-width:2px
      style C3 fill:#f3e5f5,stroke:#ce93d8,stroke-width:2px
      style D fill:#e8eaf6,stroke:#7986cb,stroke-width:2px
  </div>
  <div class="diagram-caption">
    <p>KDP processes dates by extracting components, applying appropriate transformations, and then combining them into a unified representation that captures temporal patterns.</p>
  </div>
</div>

## 🔗 Related Topics

<div class="related-topics">
  <a href="numerical-features.md" class="topic-link">
    <span class="topic-icon">🔢</span>
    <span class="topic-text">Numerical Features</span>
  </a>
  <a href="cross-features.md" class="topic-link">
    <span class="topic-icon">➕</span>
    <span class="topic-text">Cross Features</span>
  </a>
  <a href="../advanced/tabular-attention.md" class="topic-link">
    <span class="topic-icon">👁️</span>
    <span class="topic-text">Tabular Attention</span>
  </a>
  <a href="../advanced/feature-selection.md" class="topic-link">
    <span class="topic-icon">🎯</span>
    <span class="topic-text">Feature Selection</span>
  </a>
</div>

---

<div class="nav-container">
  <a href="text-features.md" class="nav-button prev">
    <span class="nav-icon">←</span>
    <span class="nav-text">Text Features</span>
  </a>
  <a href="cross-features.md" class="nav-button next">
    <span class="nav-text">Cross Features</span>
    <span class="nav-icon">→</span>
  </a>
</div>

<style>
/* Base styling */
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.6;
  color: #333;
  margin: 0;
  padding: 0;
}

/* Feature header */
.feature-header {
  background: linear-gradient(135deg, #1976d2 0%, #64b5f6 100%);
  border-radius: 10px;
  padding: 30px;
  margin: 30px 0;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  color: white;
}

.feature-title h2 {
  margin-top: 0;
  font-size: 28px;
}

.feature-title p {
  font-size: 18px;
  margin-bottom: 0;
  opacity: 0.9;
}

/* Overview card */
.overview-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px 25px;
  margin: 20px 0;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  border-left: 4px solid #1976d2;
}

.overview-card p {
  margin: 0;
  font-size: 16px;
}

/* Approaches */
.approaches-container {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.approach-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.approach-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.approach-icon {
  font-size: 2.5em;
  margin-bottom: 15px;
}

.approach-card h3 {
  margin: 0 0 10px 0;
  color: #1976d2;
}

.approach-card p {
  margin: 0;
}

/* Code containers */
.code-container {
  background-color: #f8f9fa;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  margin: 20px 0;
}

.code-container pre {
  margin: 0;
  padding: 20px;
}

/* Advanced section */
.advanced-section {
  background-color: #f8f9fa;
  border-radius: 10px;
  padding: 20px;
  margin: 30px 0;
  border-left: 4px solid #1976d2;
}

.advanced-section p {
  margin-top: 0;
}

/* Tables */
.table-container {
  margin: 30px 0;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}

.config-table {
  width: 100%;
  border-collapse: collapse;
}

.config-table th {
  background-color: #e3f2fd;
  padding: 15px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #1976d2;
}

.config-table td {
  padding: 12px 15px;
  border-bottom: 1px solid #eaecef;
}

.config-table tr:nth-child(even) {
  background-color: #f8f9fa;
}

.config-table tr:hover {
  background-color: #e3f2fd;
}

/* Power features */
.power-features {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.power-feature-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.power-feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.power-feature-card h3 {
  margin-top: 0;
  color: #1976d2;
}

/* Examples */
.examples-container {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.example-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.example-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.example-card h3 {
  margin-top: 0;
  color: #1976d2;
}

/* Pro tips */
.pro-tips-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.pro-tip-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.pro-tip-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.pro-tip-card h3 {
  margin-top: 0;
  color: #1976d2;
}

.pro-tip-card p {
  margin-bottom: 10px;
}

/* Architecture diagram */
.architecture-diagram {
  background-color: white;
  border-radius: 10px;
  padding: 20px;
  margin: 30px 0;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  text-align: center;
}

.diagram-caption {
  margin-top: 20px;
  text-align: center;
  font-style: italic;
}

/* Related topics */
.related-topics {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
  margin: 30px 0;
}

.topic-link {
  display: flex;
  align-items: center;
  padding: 10px 15px;
  background-color: #e3f2fd;
  border-radius: 8px;
  text-decoration: none;
  color: #333;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  transition: background-color 0.3s ease, transform 0.3s ease;
}

.topic-link:hover {
  background-color: #bbdefb;
  transform: translateY(-2px);
}

.topic-icon {
  font-size: 1.2em;
  margin-right: 10px;
}

/* Navigation */
.nav-container {
  display: flex;
  justify-content: space-between;
  margin: 40px 0;
}

.nav-button {
  display: flex;
  align-items: center;
  padding: 10px 15px;
  background-color: #f8f9fa;
  border-radius: 8px;
  text-decoration: none;
  color: #333;
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  transition: background-color 0.3s ease, transform 0.3s ease;
}

.nav-button:hover {
  background-color: #e3f2fd;
  transform: translateY(-2px);
}

.nav-button.prev {
  padding-left: 10px;
}

.nav-button.next {
  padding-right: 10px;
}

.nav-icon {
  font-size: 1.2em;
  margin: 0 8px;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .approaches-container,
  .power-features,
  .examples-container,
  .pro-tips-grid {
    grid-template-columns: 1fr;
  }

  .related-topics {
    flex-direction: column;
  }
}
</style>
