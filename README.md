# DSHelperTool

A lightweight helper package for data science and analysis tasks. DSHelperTool aims to reduce the hassle of common data analysis tasks by 80-90% with simple, intuitive functions.

## Installation

```bash
pip install dshelpertool
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
from dshelpertool import overview as ov

# Load your data
df = pd.read_csv("your_data.csv")

# Get a comprehensive overview
ov.quick_look(df, name="My Dataset")
```

### Clean Column Names

```python
from dshelpertool import cleaning as cl

# Standardize all column names (lowercase with underscores)
df = cl.update_col(df, standardize_col=True)

# Or rename specific columns
df = cl.update_col(df, rename_dict={"Old Name": "new_name", "Another Column": "another_column"})
```

### Handle Data Types

```python
from dshelpertool import dtypes as dt

# Convert columns to numeric
df = dt.to_numeric_cols(df, ["quantity", "price", "amount"])

# Convert columns to datetime
df = dt.to_datetime_cols(df, "transaction_date")
```

### Analyze Categorical Data

```python
from dshelpertool import overview as ov

# Get value counts for all categorical columns
result = ov.value_counts_all(df, top_n=5)

# Visualize the distributions
ov.plot_value_counts(df, columns=["category", "status"], top_n=5)
```

### Check for Skewness

```python
from dshelpertool import eda

# Check skewness of numeric columns
eda.check_skew(df, cols=["price", "quantity"])

# Visualize the distributions
eda.plot_skew(df, cols=["price", "quantity"])
```

## Development

### Setting up the Development Environment

```bash
# Clone the repository
git clone https://github.com/uumap/dshelpertool.git
cd dshelpertool

# Install in development mode with development dependencies
pip install -e .[dev]

# Run tests
python -m pytest
```

### GitHub Actions Workflows

This project uses GitHub Actions for continuous integration and deployment:

1. **Python Tests** (`python-test.yml`): Runs on every push to main and pull request
   - Runs tests on multiple Python versions
   - Performs linting with flake8
   - Checks code formatting with black
   - Checks import order with isort
   - Reports test coverage

2. **Publish Python Package** (`python-publish.yml`): Runs when a new release is created
   - Tests the package on multiple Python versions
   - Builds the package
   - Publishes to PyPI

To manually trigger the test workflow, go to the Actions tab in your GitHub repository and select "Python Tests" workflow, then click "Run workflow".

## Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`pytest`)
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# DSHelperTool Documentation

## Overview
DSHelperTool is a lightweight helper package for data science and analysis tasks. Below is a comprehensive list of functions available in the package, along with their use cases and examples.

---

## Cleaning Module

### `drop_cols`
**Description:** Drop specified columns from a DataFrame.
**Parameters:**
- `df`: Input DataFrame.
- `cols`: List of columns to drop.
**Example:**
```python
from dshelpertool.cleaning import drop_cols
cleaned_df = drop_cols(df, cols=['unnecessary_column'])
```

### `update_col`
**Description:** Update column names or standardize them.
**Parameters:**
- `df`: Input DataFrame.
- `rename_dict`: Dictionary for renaming columns.
- `standardize_col`: Boolean to standardize column names.
**Example:**
```python
from dshelpertool.cleaning import update_col
updated_df = update_col(df, rename_dict={'OldName': 'NewName'})
```

---

## EDA Module

### `check_skew`
**Description:** Check skewness of numeric columns.
**Parameters:**
- `df`: Input DataFrame.
- `cols`: List of columns to check skewness.
**Example:**
```python
from dshelpertool.eda import check_skew
check_skew(df, cols=['column1', 'column2'])
```

### `plot_skew`
**Description:** Plot histograms to visualize skewness.
**Parameters:**
- `df`: Input DataFrame.
- `cols`: List of columns to plot.
- `bins`: Number of bins for histograms.
**Example:**
```python
from dshelpertool.eda import plot_skew
plot_skew(df, cols=['column1', 'column2'], bins=20)
```

---

## Visualization Module

### `plot_distributions`
**Description:** Plot distributions of numeric columns.
**Parameters:**
- `df`: Input DataFrame.
- `cols`: List of columns to plot.
- `bins`: Number of bins for histograms.
**Example:**
```python
from dshelpertool.viz import plot_distributions
plot_distributions(df, cols=['column1', 'column2'], bins=30)
```

### `plot_categorical`
**Description:** Plot bar charts for categorical columns.
**Parameters:**
- `df`: Input DataFrame.
- `cols`: List of columns to plot.
- `top_n`: Number of top categories to display.
**Example:**
```python
from dshelpertool.viz import plot_categorical
plot_categorical(df, cols=['category_column'], top_n=5)
```

### `plot_correlation`
**Description:** Plot correlation matrix for numeric columns.
**Parameters:**
- `df`: Input DataFrame.
- `method`: Correlation method ('pearson', 'kendall', 'spearman').
**Example:**
```python
from dshelpertool.viz import plot_correlation
plot_correlation(df, method='pearson')
```

---

## Statistics Module

### `correlation_matrix`
**Description:** Calculate and visualize the correlation matrix.
**Parameters:**
- `df`: Input DataFrame.
- `method`: Correlation method ('pearson', 'kendall', 'spearman').
**Example:**
```python
from dshelpertool.stats import correlation_matrix
correlation_matrix(df, method='spearman')
```

### `hypothesis_test`
**Description:** Perform hypothesis tests to compare groups.
**Parameters:**
- `df`: Input DataFrame.
- `group_col`: Column containing group labels.
- `value_col`: Column containing values to compare.
**Example:**
```python
from dshelpertool.stats import hypothesis_test
hypothesis_test(df, group_col='group', value_col='value', test_type='ttest')
```

---

## Overview Module

### `quick_look`
**Description:** Get a quick overview of a DataFrame.
**Parameters:**
- `df`: Input DataFrame.
- `name`: Name of the DataFrame.
**Example:**
```python
from dshelpertool.overview import quick_look
quick_look(df, name='My Dataset')
```

### `get_duplicates`
**Description:** Identify duplicate rows in a DataFrame.
**Parameters:**
- `df`: Input DataFrame.
- `subset`: Columns to consider for identifying duplicates.
**Example:**
```python
from dshelpertool.overview import get_duplicates
duplicates = get_duplicates(df, subset=['column1'])
```

---

## Report Module

### `export_to_excel`
**Description:** Export DataFrame to an Excel file.
**Parameters:**
- `df`: Input DataFrame.
- `filename`: Name of the Excel file.
**Example:**
```python
from dshelpertool.report import export_to_excel
export_to_excel(df, filename='output.xlsx')
```

---

This documentation provides a detailed overview of the functions available in DSHelperTool. For more examples and advanced usage, refer to the examples folder in the project.
