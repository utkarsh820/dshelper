# DSHelper

A lightweight helper package for data science and analysis tasks. DSHelper aims to reduce the hassle of common data analysis tasks by 80-90% with simple, intuitive functions.

## Installation

```bash
pip install dshelper
```

## Features

- **Quick Data Overview**: Get a comprehensive summary of your DataFrame with a single function call
- **Data Cleaning**: Easily standardize column names, handle missing values, and fix data types
- **Exploratory Data Analysis**: Analyze distributions, correlations, and patterns in your data
- **Visualization**: Create informative plots with minimal code
- **Data Type Handling**: Convert columns to appropriate data types with smart error handling

## Usage Examples

### Quick Overview of a DataFrame

```python
import pandas as pd
from dshelper import overview as ov

# Load your data
df = pd.read_csv("your_data.csv")

# Get a comprehensive overview
ov.quick_look(df, name="My Dataset")
```

### Clean Column Names

```python
from dshelper import cleaning as cl

# Standardize all column names (lowercase with underscores)
df = cl.update_col(df, standardize_col=True)

# Or rename specific columns
df = cl.update_col(df, rename_dict={"Old Name": "new_name", "Another Column": "another_column"})
```

### Handle Data Types

```python
from dshelper import dtypes as dt

# Convert columns to numeric
df = dt.to_numeric_cols(df, ["quantity", "price", "amount"])

# Convert columns to datetime
df = dt.to_datetime_cols(df, "transaction_date")
```

### Analyze Categorical Data

```python
from dshelper import overview as ov

# Get value counts for all categorical columns
result = ov.value_counts_all(df, top_n=5)

# Visualize the distributions
ov.plot_value_counts(df, columns=["category", "status"], top_n=5)
```

### Check for Skewness

```python
from dshelper import eda

# Check skewness of numeric columns
eda.check_skew(df, cols=["price", "quantity"])

# Visualize the distributions
eda.plot_skew(df, cols=["price", "quantity"])
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
