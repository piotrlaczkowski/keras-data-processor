# 📅 Date Features

> Extract powerful patterns from temporal data like timestamps, dates, and time series.

## 📋 Quick Overview

Date features transform timestamps and dates into ML-ready representations. KDP automatically extracts useful components like day of week, month, and year, and can even detect cyclical patterns and seasonality.

## 🚀 Basic Usage

The simplest way to define date features is with the `FeatureType` enum:

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

## 🧠 Advanced Configuration

For more control, use the `DateFeature` class:

```python
from kdp.features import DateFeature

features = {
    # Transaction date with full components
    "transaction_date": DateFeature(
        name="transaction_date",
        feature_type=FeatureType.DATE,
        date_format="%Y-%m-%d",       # Input format
        add_day_of_week=True,         # Add day of week (Mon, Tue, etc.)
        add_day_of_month=True,        # Add day of month (1-31)
        add_month=True,               # Add month (1-12)
        add_year=True,                # Add year
        add_quarter=True,             # Add quarter (1-4)
        cyclical_encoding=True        # Use sine/cosine encoding
    ),

    # User signup date with seasonality
    "signup_date": DateFeature(
        name="signup_date",
        feature_type=FeatureType.DATE,
        add_season=True,              # Add season (Winter, Spring, etc.)
        add_is_weekend=True,          # Add weekend indicator
        add_age=True,                 # Add age of date in days
        reference_date="2020-01-01"   # Reference for age calculation
    ),

    # Event timestamp
    "event_timestamp": DateFeature(
        name="event_timestamp",
        feature_type=FeatureType.DATE,
        is_timestamp=True,            # Parse as timestamp
        timestamp_unit="s",           # Seconds
        add_hour=True,                # Add hour of day
        add_minute=True,              # Add minute
        timezone="UTC"                # Specify timezone
    )
}
```

## ⚙️ Key Configuration Options

| Parameter | Description | Default | Suggested Range |
|-----------|-------------|---------|----------------|
| `date_format` | Input date format | Auto-detect | Standard format strings |
| `is_timestamp` | Whether input is timestamp | `False` | `True`/`False` |
| `timestamp_unit` | Unit for timestamps | "s" | "s", "ms", "us", "ns" |
| `add_year` | Include year component | `True` | `True`/`False` |
| `add_month` | Include month component | `True` | `True`/`False` |
| `add_day_of_month` | Include day component | `True` | `True`/`False` |
| `add_day_of_week` | Include day of week | `True` | `True`/`False` |
| `add_hour` | Include hour component | `False` | `True`/`False` |
| `add_minute` | Include minute component | `False` | `True`/`False` |
| `add_second` | Include second component | `False` | `True`/`False` |
| `add_quarter` | Include quarter | `False` | `True`/`False` |
| `add_season` | Include season | `False` | `True`/`False` |
| `add_is_weekend` | Include weekend flag | `False` | `True`/`False` |
| `add_age` | Include age in days | `False` | `True`/`False` |
| `reference_date` | Reference for age | Current date | Any date string |
| `cyclical_encoding` | Use sine/cosine encoding | `False` | `True`/`False` |
| `timezone` | Timezone for parsing | `None` | "UTC", "America/New_York", etc. |

## 🔥 Power Features

### Cyclical Encoding

Properly represent cyclical time components:

```python
# Better representation of cyclical features
from kdp.features import DateFeature

features = {
    "purchase_date": DateFeature(
        name="purchase_date",
        feature_type=FeatureType.DATE,
        add_hour=True,
        add_day_of_week=True,
        add_month=True,
        cyclical_encoding=True         # Represent cyclical nature
    )
}
```

### Age and Time Deltas

Calculate time since or until a reference date:

```python
# Calculate customer lifetime
from kdp.features import DateFeature

features = {
    "signup_date": DateFeature(
        name="signup_date",
        feature_type=FeatureType.DATE,
        add_age=True,                  # Add age feature
        reference_date="2023-01-01"    # Reference point
    )
}
```

## 💼 Real-World Examples

### E-commerce Purchase Analysis

```python
# Analyze purchase patterns over time
preprocessor = PreprocessingModel(
    features_specs={
        "purchase_date": DateFeature(
            name="purchase_date",
            feature_type=FeatureType.DATE,
            add_day_of_week=True,     # Capture day-of-week patterns
            add_month=True,           # Capture monthly patterns
            add_is_weekend=True,      # Weekend vs weekday
            add_quarter=True,         # Quarterly patterns
            cyclical_encoding=True    # Proper cyclical representation
        ),
        "user_signup_date": DateFeature(
            name="user_signup_date",
            feature_type=FeatureType.DATE,
            add_age=True,             # How long customer has been registered
            reference_date="now"      # Relative to current date
        )
    }
)
```

### Time Series Forecasting

```python
# Time series feature extraction
preprocessor = PreprocessingModel(
    features_specs={
        "timestamp": DateFeature(
            name="timestamp",
            feature_type=FeatureType.DATE,
            is_timestamp=True,
            timestamp_unit="s",
            add_hour=True,            # Hourly patterns
            add_day_of_week=True,     # Weekly patterns
            add_month=True,           # Monthly patterns
            add_year=True,            # Yearly trends
            cyclical_encoding=True    # Sine/cosine encoding
        ),
        # Additional features...
        "value": FeatureType.FLOAT_RESCALED
    },
    # Enable temporal attention
    tabular_attention=True,
    tabular_attention_heads=4
)
```

## 💡 Pro Tips

1. **Choose the Right Components**
   - Only include time components relevant to your problem
   - For daily data, focus on day of week, month, quarter
   - For hourly data, include hour of day and day of week
   - For long-term patterns, include month, quarter, and year

2. **Always Use Cyclical Encoding**
   - Essential for cyclic time components (hour, day, month)
   - Prevents artificial boundaries (Dec 31 → Jan 1)
   - Creates smooth transitions between time periods

3. **Time Zone Handling**
   - Always specify a timezone for consistent processing
   - Use UTC for global applications
   - Use local timezones for region-specific patterns

4. **Combine with Feature Crosses**
   - Cross day of week with hour for weekly patterns
   - Cross month with year for seasonal year-over-year patterns
   - Example: `feature_crosses=[("day_of_week", "hour", 8)]`

## 🔗 Related Topics

- [Feature Crosses](cross-features.md) - Model interactions between features
- [Tabular Attention](../advanced/tabular-attention.md) - Learn temporal patterns
- [Feature Selection](../advanced/feature-selection.md) - Find important time components

---

<div class="prev-next">
  <a href="text-features.md" class="prev">← Text Features</a>
  <a href="cross-features.md" class="next">Cross Features →</a>
</div>

<style>
.prev-next {
  display: flex;
  justify-content: space-between;
  margin-top: 40px;
}
.prev-next a {
  padding: 10px 15px;
  background-color: #f1f1f1;
  border-radius: 5px;
  text-decoration: none;
  color: #333;
}
.prev-next a:hover {
  background-color: #ddd;
}
</style>
