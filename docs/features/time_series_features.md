# ⏱️ Time Series Features

<div class="feature-header">
  <div class="feature-title">
    <h2>Time Series Features in KDP</h2>
    <p>Transform temporal data with powerful lag features, moving averages, differencing, and rolling statistics.</p>
  </div>
</div>

## 📋 Overview

<div class="overview-card">
  <p>Time series features enable processing of chronological data by creating transformations that capture temporal patterns and relationships. KDP provides specialized layers for common time series operations that maintain data ordering while enabling advanced machine learning on sequential data.</p>
</div>

## 🚀 Types of Time Series Transformations

<div class="table-container">
  <table class="features-table">
    <thead>
      <tr>
        <th>Transformation</th>
        <th>Purpose</th>
        <th>Example</th>
        <th>When to Use</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>Lag Features</code></td>
        <td>Create features from past values</td>
        <td>Yesterday's sales, last week's sales</td>
        <td>When past values help predict future ones</td>
      </tr>
      <tr>
        <td><code>Rolling Statistics</code></td>
        <td>Compute statistics over windows</td>
        <td>7-day average, 30-day standard deviation</td>
        <td>When trends or volatility matter</td>
      </tr>
      <tr>
        <td><code>Differencing</code></td>
        <td>Calculate changes between values</td>
        <td>Day-over-day change in price</td>
        <td>When changes are more important than absolute values</td>
      </tr>
      <tr>
        <td><code>Moving Averages</code></td>
        <td>Smooth data over time</td>
        <td>7-day, 14-day, 28-day moving averages</td>
        <td>When you need to reduce noise and focus on trends</td>
      </tr>
    </tbody>
  </table>
</div>

## 📝 Basic Usage

<div class="code-container">

```python
from kdp import TimeSeriesFeature

# Create a time series feature for daily sales data
sales_ts = TimeSeriesFeature(
    name="sales",
    # Sort by date column to ensure chronological order
    sort_by="date",
    # Group by store to handle multiple time series
    group_by="store_id",
    # Create lag features for yesterday, last week, and two weeks ago
    lag_config={
        "lag_indices": [1, 7, 14],
        "drop_na": True,
        "fill_value": 0.0,
        "keep_original": True
    }
)
```

</div>

## 🧠 Advanced Configuration

<div class="advanced-section">
  <p>For comprehensive time series processing, configure multiple transformations in a single feature:</p>

  <div class="code-container">

```python
from kdp import TimeSeriesFeature, Processor

# Complete time series configuration with multiple transformations
sales_feature = TimeSeriesFeature(
    name="sales",
    # Data ordering configuration
    sort_by="date",                           # Column to sort by
    sort_ascending=True,                      # Sort chronologically
    group_by="store_id",                      # Group by store

    # Lag feature configuration
    lag_config={
        "lag_indices": [1, 7, 14, 28],        # Previous day, week, 2 weeks, 4 weeks
        "drop_na": True,                      # Remove rows with insufficient history
        "fill_value": 0.0,                    # Value for missing lags if drop_na=False
        "keep_original": True                 # Include original values
    },

    # Rolling statistics configuration
    rolling_stats_config={
        "window_size": 7,                     # 7-day rolling window
        "statistics": ["mean", "std", "min", "max"],  # Statistics to compute
        "window_stride": 1,                   # Move window by 1 time step
        "drop_na": True                       # Remove rows with insufficient history
    },

    # Differencing configuration
    differencing_config={
        "order": 1,                           # First-order differencing (t - (t-1))
        "drop_na": True,                      # Remove rows with insufficient history
        "fill_value": 0.0,                    # Value for missing diffs if drop_na=False
        "keep_original": True                 # Include original values
    },

    # Moving average configuration
    moving_average_config={
        "periods": [7, 14, 28],               # Weekly, bi-weekly, monthly averages
        "drop_na": True,                      # Remove rows with insufficient history
        "pad_value": 0.0                      # Value for padding if drop_na=False
    }
)

# Create processor with time series feature
processor = Processor(
    features=[sales_feature],
    target="next_day_sales"  # Target variable to predict
)
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
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>sort_by</code></td>
        <td>Column used for ordering data</td>
        <td>Required</td>
        <td>Typically a date or timestamp column</td>
      </tr>
      <tr>
        <td><code>sort_ascending</code></td>
        <td>Sort direction</td>
        <td>True</td>
        <td>True for oldest→newest, False for newest→oldest</td>
      </tr>
      <tr>
        <td><code>group_by</code></td>
        <td>Column for grouping multiple series</td>
        <td>None</td>
        <td>Optional, for handling multiple related series</td>
      </tr>
      <tr>
        <td><code>lag_indices</code></td>
        <td>Time steps to look back</td>
        <td>None</td>
        <td>List of integers, e.g. [1, 7] for yesterday and last week</td>
      </tr>
      <tr>
        <td><code>window_size</code></td>
        <td>Size of rolling window</td>
        <td>7</td>
        <td>Number of time steps to include in window</td>
      </tr>
      <tr>
        <td><code>statistics</code></td>
        <td>Rolling statistics to compute</td>
        <td>["mean"]</td>
        <td>Options: "mean", "std", "min", "max", "sum"</td>
      </tr>
      <tr>
        <td><code>order</code></td>
        <td>Differencing order</td>
        <td>1</td>
        <td>1=first difference, 2=second difference, etc.</td>
      </tr>
      <tr>
        <td><code>periods</code></td>
        <td>Moving average periods</td>
        <td>None</td>
        <td>List of integers, e.g. [7, 30] for weekly and monthly</td>
      </tr>
      <tr>
        <td><code>drop_na</code></td>
        <td>Remove rows with insufficient history</td>
        <td>True</td>
        <td>Set to False to keep all rows with padding</td>
      </tr>
    </tbody>
  </table>
</div>

## 💡 Powerful Features

<div class="power-features">
  <div class="power-feature-card">
    <h3>🔄 Automatic Data Ordering</h3>
    <p>KDP automatically handles the correct ordering of time series data:</p>
    <div class="code-container">

```python
from kdp import TimeSeriesFeature, Processor

# Define a time series feature with automatic ordering
sales_ts = TimeSeriesFeature(
    name="sales",
    # Specify which column contains timestamps/dates
    sort_by="timestamp",
    # Sort in ascending order (oldest first)
    sort_ascending=True,
    # Group by store to create separate series per store
    group_by="store_id",
    # Simple lag configuration
    lag_config={"lag_indices": [1, 7]}
)

# Even with shuffled data, KDP will correctly order the features
processor = Processor(
    features=[sales_ts],
    target="target"
)

# The preprocessor handles ordering before applying transformations
preprocessor = processor.build_preprocessor(shuffled_data)
```

    </div>
  </div>

  <div class="power-feature-card">
    <h3>📊 Handling Multiple Time Series</h3>
    <p>Process multiple related time series with the <code>group_by</code> parameter:</p>
    <div class="code-container">

```python
# Process sales data from multiple stores
multi_store_sales = TimeSeriesFeature(
    name="sales",
    # Sort by date for chronological ordering
    sort_by="date",
    # Group by store_id to handle each store separately
    group_by="store_id",
    # Configure lag features
    lag_config={
        "lag_indices": [1, 7, 14],  # Yesterday, last week, two weeks ago
        "keep_original": True
    }
)

# Each store's sales will be processed as a separate time series
# Lags for Store A will only use Store A's history
# Lags for Store B will only use Store B's history

# Use in processor
processor = Processor(
    features=[multi_store_sales],
    target="tomorrow_sales"
)
```

    </div>
  </div>

  <div class="power-feature-card">
    <h3>🔍 Date/Time Feature Extraction</h3>
    <p>Extract useful components from datetime columns:</p>
    <div class="code-container">

```python
from kdp import TimeSeriesFeature, DatetimeFeature

# Extract time components from the date column
date_components = DatetimeFeature(
    name="date",
    # Extract useful time components
    extracted_components=["day_of_week", "month", "quarter", "year"],
    # Optionally create cyclical encoding for day of week and month
    cyclical_encoding=True
)

# Combine with time series features
sales_ts = TimeSeriesFeature(
    name="sales",
    sort_by="date",
    lag_config={"lag_indices": [1, 7, 14]}
)

# Use both features in processor
processor = Processor(
    features=[date_components, sales_ts],
    target="target"
)

# The date components (e.g., is_weekend, month) help the model
# learn seasonal patterns, while the lags capture recent trends
```

    </div>
  </div>

  <div class="power-feature-card">
    <h3>⚖️ Batch Processing for Time Series</h3>
    <p>KDP efficiently handles time series data in batches while maintaining correct ordering:</p>
    <div class="code-container">

```python
import tensorflow as tf
from kdp import TimeSeriesFeature, Processor

# Define time series feature
sales_feature = TimeSeriesFeature(
    name="sales",
    sort_by="date",
    group_by="store_id",
    lag_config={"lag_indices": [1, 7]}
)

# Build processor and preprocessor
processor = Processor(features=[sales_feature], target="target")
preprocessor = processor.build_preprocessor(train_data)

# Create a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices(
    {"sales": test_data["sales"].values,
     "date": test_data["date"].values,
     "store_id": test_data["store_id"].values}
)

# Batch the data - KDP will maintain proper time series ordering
batched_dataset = dataset.batch(32)

# Apply preprocessor - each batch will be correctly processed
processed_dataset = batched_dataset.map(preprocessor)

# The time series ordering is maintained within each group (store_id)
# even when processing in batches
```

    </div>
  </div>
</div>

## 🔧 Real-World Examples

<div class="examples-container">
  <div class="example-card">
    <h3>📈 Retail Sales Forecasting</h3>
    <div class="code-container">

```python
from kdp import TimeSeriesFeature, DatetimeFeature, CategoricalFeature, Processor

# Define features for sales forecasting
features = [
    # Time series features for sales data
    TimeSeriesFeature(
        name="sales",
        sort_by="date",
        group_by="store_id",
        # Recent sales and same period in previous years
        lag_config={
            "lag_indices": [1, 2, 3, 7, 14, 28, 365, 365+7],
            "keep_original": True
        },
        # Weekly and monthly trends
        rolling_stats_config={
            "window_size": 7,
            "statistics": ["mean", "std", "min", "max"]
        },
        # Day-over-day changes
        differencing_config={
            "order": 1,
            "keep_original": True
        },
        # Weekly, monthly, quarterly smoothing
        moving_average_config={
            "periods": [7, 30, 90]
        }
    ),

    # Date component extraction
    DatetimeFeature(
        name="date",
        extracted_components=["day_of_week", "day_of_month", "month", "is_weekend", "is_holiday"],
        cyclical_encoding=True  # Use sine/cosine encoding for cyclical features
    ),

    # Store features
    CategoricalFeature(
        name="store_id",
        embedding_dim=8
    ),

    # Product category
    CategoricalFeature(
        name="product_category",
        embedding_dim=8
    )
]

# Create processor
sales_forecaster = Processor(
    features=features,
    target="next_day_sales"
)

# Build preprocessor
preprocessor = sales_forecaster.build_preprocessor(train_data)
```

    </div>
  </div>

  <div class="example-card">
    <h3>📊 Stock Price Analysis</h3>
    <div class="code-container">

```python
from kdp import TimeSeriesFeature, Processor

# Define time series features for stock data
stock_features = [
    # Price features
    TimeSeriesFeature(
        name="close_price",
        sort_by="date",
        sort_ascending=True,
        # Previous days' prices
        lag_config={
            "lag_indices": [1, 2, 3, 5, 10, 20],
            "keep_original": True
        },
        # Price momentum
        differencing_config={
            "order": 1,
            "keep_original": True
        },
        # Moving averages for trend identification
        moving_average_config={
            "periods": [5, 10, 20, 50, 200]
        }
    ),

    # Volume features with lags
    TimeSeriesFeature(
        name="volume",
        sort_by="date",
        lag_config={
            "lag_indices": [1, 2, 3, 5],
            "keep_original": True
        },
        # Volume trends
        rolling_stats_config={
            "window_size": 5,
            "statistics": ["mean", "max"]
        }
    ),

    # Volatility calculation
    TimeSeriesFeature(
        name="high_low_range",  # high_price - low_price
        sort_by="date",
        lag_config={
            "lag_indices": [1, 2, 3, 5],
            "keep_original": True
        },
        rolling_stats_config={
            "window_size": 10,
            "statistics": ["mean", "std"]  # Volatility measures
        }
    )
]

# Create processor for stock price prediction
stock_predictor = Processor(
    features=stock_features,
    target="next_day_return"  # Percentage return for next day
)
```

    </div>
  </div>

  <div class="example-card">
    <h3>🏥 Patient Monitoring</h3>
    <div class="code-container">

```python
from kdp import TimeSeriesFeature, CategoricalFeature, NumericalFeature, Processor

# Define features for patient monitoring
features = [
    # Vital signs as time series
    TimeSeriesFeature(
        name="heart_rate",
        sort_by="timestamp",
        group_by="patient_id",
        # Recent measurements
        lag_config={
            "lag_indices": [1, 2, 3, 6, 12, 24],  # Hours back
            "keep_original": True
        },
        # Short and long-term trends
        rolling_stats_config={
            "window_size": 6,  # 6-hour window
            "statistics": ["mean", "std", "min", "max"]
        }
    ),

    # Blood pressure
    TimeSeriesFeature(
        name="blood_pressure",
        sort_by="timestamp",
        group_by="patient_id",
        lag_config={
            "lag_indices": [1, 6, 12, 24]
        },
        rolling_stats_config={
            "window_size": 12,  # 12-hour window
            "statistics": ["mean", "std"]
        }
    ),

    # Body temperature
    TimeSeriesFeature(
        name="temperature",
        sort_by="timestamp",
        group_by="patient_id",
        lag_config={
            "lag_indices": [1, 2, 6, 12]
        },
        rolling_stats_config={
            "window_size": 6,
            "statistics": ["mean", "min", "max"]
        }
    ),

    # Patient demographics
    NumericalFeature(name="age"),
    CategoricalFeature(name="gender"),
    CategoricalFeature(
        name="diagnosis",
        embedding_dim=16
    )
]

# Create processor for patient risk prediction
patient_monitor = Processor(
    features=features,
    target="risk_score"  # Risk score for the next 24 hours
)
```

    </div>
  </div>
</div>

## 💎 Pro Tips

<div class="pro-tips-grid">
  <div class="pro-tip-card">
    <h3>🔍 Choose Meaningful Lag Features</h3>
    <p>When selecting lag indices, consider domain knowledge about your data:</p>
    <ul>
      <li>For daily data: include 1 (yesterday), 7 (last week), and 30 (last month)</li>
      <li>For hourly data: include 1, 24 (same hour yesterday), 168 (same hour last week)</li>
      <li>For seasonal patterns: include 365 (same day last year) for annual data</li>
      <li>For quarterly financials: include 1, 4 (same quarter last year)</li>
    </ul>
    <p>This captures daily, weekly, and seasonal patterns that might exist in your data.</p>
  </div>

  <div class="pro-tip-card">
    <h3>📊 Combine Multiple Transformations</h3>
    <p>Different time series transformations capture different aspects of your data:</p>
    <ul>
      <li><strong>Lag features:</strong> Capture direct dependencies on past values</li>
      <li><strong>Rolling statistics:</strong> Capture trends and volatility</li>
      <li><strong>Differencing:</strong> Captures changes and removes trend</li>
      <li><strong>Moving averages:</strong> Smooths noise and highlights trends</li>
    </ul>
    <p>Using these together creates a rich feature set that captures various temporal patterns.</p>
  </div>

  <div class="pro-tip-card">
    <h3>⚠️ Handle the Cold Start Problem</h3>
    <p>New time series may not have enough history for lag features:</p>
    <div class="code-container">

```python
# Gracefully handle new entities with insufficient history
sales_ts = TimeSeriesFeature(
    name="sales",
    sort_by="date",
    group_by="store_id",
    lag_config={
        "lag_indices": [1, 7],
        "drop_na": False,     # Keep rows with missing lags
        "fill_value": 0.0     # Use 0 for missing values
    }
)

# Alternative: Add a "data_available" feature
has_history = "df['store_age'] > 7"  # Custom logic

# Combine with other features
```

   </div>
  </div>

  <div class="pro-tip-card">
    <h3>🔄 Preprocessing Order Matters</h3>
    <p>The order of operations for time series preprocessing is important:</p>
    <ol>
      <li>First, group data (if using <code>group_by</code>)</li>
      <li>Then sort within each group (by <code>sort_by</code>)</li>
      <li>Finally, apply transformations in order: differencing, lag features, rolling stats, moving averages</li>
    </ol>
    <p>KDP handles this automatically, but it's good to understand the sequence.</p>
  </div>
</div>

## 📊 Understanding Time Series Layers

<div class="architecture-diagram">
  <div class="mermaid">
    graph TD
      A[Raw Time Series Data] -->|Sort & Group| B[Ordered Data]
      B -->|Apply Transformations| C[Processed Features]
      C -->|Feed to Model| D[ML Model]

      subgraph "Time Series Transformations"
        E[Lag Features]
        F[Rolling Statistics]
        G[Differencing]
        H[Moving Averages]
      end

      B --> E --> C
      B --> F --> C
      B --> G --> C
      B --> H --> C

      style A fill:#f9f9f9,stroke:#ccc,stroke-width:2px
      style B fill:#e1f5fe,stroke:#4fc3f7,stroke-width:2px
      style C fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px
      style D fill:#f3e5f5,stroke:#ce93d8,stroke-width:2px
  </div>
  <div class="diagram-caption">
    <p>The diagram shows how time series data flows through the KDP preprocessing pipeline. First, data is sorted and grouped, then various transformations are applied in parallel, and finally, the processed features are fed to the machine learning model.</p>
  </div>
</div>

## 🔗 Related Topics

<div class="related-topics">
  <a href="categorical-features.md" class="topic-link">
    <span class="topic-icon">🏷️</span>
    <span class="topic-text">Categorical Features</span>
  </a>
  <a href="numerical-features.md" class="topic-link">
    <span class="topic-icon">🔢</span>
    <span class="topic-text">Numerical Features</span>
  </a>
  <a href="../advanced/custom-layers.md" class="topic-link">
    <span class="topic-icon">🧩</span>
    <span class="topic-text">Custom Preprocessing Layers</span>
  </a>
  <a href="../examples/time-series-forecasting.md" class="topic-link">
    <span class="topic-icon">📈</span>
    <span class="topic-text">Time Series Forecasting Examples</span>
  </a>
</div>

---

<div class="nav-container">
  <a href="text-features.md" class="nav-button prev">
    <span class="nav-icon">←</span>
    <span class="nav-text">Text Features</span>
  </a>
  <a href="image-features.md" class="nav-button next">
    <span class="nav-text">Image Features</span>
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
  background: linear-gradient(135deg, #43a047 0%, #1de9b6 100%);
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
  border-left: 4px solid #43a047;
}

.overview-card p {
  margin: 0;
  font-size: 16px;
}

/* Tables */
.table-container {
  margin: 30px 0;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}

.features-table, .config-table {
  width: 100%;
  border-collapse: collapse;
}

.features-table th, .config-table th {
  background-color: #f0fff0;
  padding: 15px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #43a047;
}

.features-table td, .config-table td {
  padding: 12px 15px;
  border-bottom: 1px solid #eaecef;
}

.features-table tr:nth-child(even), .config-table tr:nth-child(even) {
  background-color: #f8f9fa;
}

.features-table tr:hover, .config-table tr:hover {
  background-color: #f0fff0;
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
  border-left: 4px solid #43a047;
}

.advanced-section p {
  margin-top: 0;
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
  color: #43a047;
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
  color: #43a047;
}

/* Pro tips */
.pro-tips-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
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
  color: #43a047;
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
  background-color: #f0fff0;
  border-radius: 8px;
  text-decoration: none;
  color: #333;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  transition: background-color 0.3s ease, transform 0.3s ease;
}

.topic-link:hover {
  background-color: #e0f2e0;
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
  background-color: #f0fff0;
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
