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
- **Statistical Analysis**: Perform common statistical tests and analyses
- **Reporting**: Generate detailed reports in various formats

## Usage Examples

### Overview Module

```python
from dshelpertool import overview as ov

# Get comprehensive DataFrame overview
summary = ov.quick_look(df, name="My Dataset")

# Analyze value counts for categorical columns
counts = ov.value_counts_all(df, top_n=5)

# Find duplicate rows
duplicates = ov.get_duplicates(df, subset=['column1', 'column2'])
```

### Cleaning Module

```python
from dshelpertool import cleaning as cl

# Drop unnecessary columns
df = cl.drop_cols(df, cols=['unnecessary_column'])

# Standardize column names (lowercase with underscores)
df = cl.update_col(df, standardize_col=True)

# Rename specific columns
df = cl.update_col(df, rename_dict={
    "Old Name": "new_name", 
    "Another Column": "another_column"
})
```

### Data Types Module

```python
from dshelpertool import dtypes as dt

# Convert columns to numeric type
df = dt.to_numeric_cols(df, ["quantity", "price", "amount"])

# Convert columns to datetime
df = dt.to_datetime_cols(df, ["transaction_date", "delivery_date"])

# Analyze memory usage and get optimization suggestions
memory_info = dt.memory_usage(df, deep=True)
```

### EDA (Exploratory Data Analysis) Module

```python
from dshelpertool import eda

# Check skewness of numeric columns
skew_stats = eda.check_skew(df, cols=["price", "quantity"])

# Visualize skewness
eda.plot_skew(df, cols=["price", "quantity"])

# Handle missing values
df = eda.fill_missing(df, method="mean", cols=["price", "quantity"])
```

### Statistics Module

```python
from dshelpertool import stats

# Get enhanced descriptive statistics
desc_stats = stats.describe_all(df)

# Calculate correlation matrix
corr = stats.correlation_matrix(df, method='pearson')

# Perform hypothesis testing
test_results = stats.hypothesis_test(
    df, 
    group_col='group', 
    value_col='measurement'
)

# Detect outliers
outliers = stats.outlier_detection(
    df, 
    cols=['price', 'quantity'], 
    method='iqr'
)
```

### Visualization Module

```python
from dshelpertool import viz

# Plot distributions of numeric columns
viz.plot_distributions(df, cols=['price', 'quantity'], bins=30)

# Create categorical plots
viz.plot_categorical(df, cols=['category', 'status'], top_n=5)

# Visualize correlation matrix
viz.plot_correlation(df, method='pearson')
```

### Reporting Module

```python
from dshelpertool import report

# Export DataFrame to Excel with formatting
report.export_to_excel(df, filename='analysis_report.xlsx')

# Generate data dictionary
dictionary = report.generate_data_dictionary(df)

# Create comprehensive summary report
report.generate_summary_report(
    df,
    title="Analysis Report",
    output_format="markdown",
    include_plots=True
)
```

### Command Line Interface

The package also provides a CLI for quick analysis:

```bash
# Get quick overview of a dataset
dshelpertool overview data.csv --name "My Dataset"

# Generate a report
dshelpertool report data.csv --format markdown --output report.md

# Check data types and memory usage
dshelpertool dtypes data.csv
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
   - Runs tests on multiple Python versions (3.8-3.11)
   - Performs linting with flake8
   - Checks code formatting with black
   - Checks import order with isort
   - Reports test coverage

2. **Publish Python Package** (`python-publish.yml`): Runs when a new release is created
   - Tests the package on multiple Python versions
   - Builds the package
   - Publishes to PyPI

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

