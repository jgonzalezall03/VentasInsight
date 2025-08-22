# Sales Analytics Dashboard

## Overview

This is a comprehensive sales analytics dashboard built with Streamlit that provides data visualization, team performance analysis, and sales forecasting capabilities. The application allows users to upload Excel files containing sales data and automatically processes them to generate insights through interactive charts, KPI metrics, and predictive analytics. The dashboard is designed for sales managers and teams to analyze individual and collective performance, identify trends, and make data-driven forecasting decisions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework providing an interactive dashboard interface
- **Layout**: Multi-page application with sidebar navigation and wide layout configuration
- **Page Structure**: Main dashboard (`app.py`) with separate pages for team analysis, individual analysis, and forecasting
- **Visualization Library**: Plotly for interactive charts and graphs
- **State Management**: Streamlit session state for maintaining data processor and dataframe across page interactions

### Data Processing Architecture
- **Data Ingestion**: Excel file upload functionality with support for .xlsx and .xls formats
- **Data Processing**: Automatic column detection using pattern matching for dates, amounts, salespeople, products, customers, and regions
- **Data Cleaning**: Built-in data validation, type conversion, and preprocessing pipeline
- **Column Mapping**: Smart detection of common sales data columns (fecha/date, monto/amount, vendedor/salesperson)

### Analytics Architecture
- **KPI Calculations**: Real-time computation of sales metrics including totals, averages, transaction counts, and team statistics
- **Time Series Analysis**: Date-based filtering and aggregation for temporal analysis
- **Team Performance**: Comparative analysis across salespeople with ranking and statistical measures
- **Individual Analysis**: Detailed performance breakdown for specific salespeople including product analysis

### Forecasting Architecture
- **Machine Learning**: Scikit-learn linear regression model for sales prediction
- **Time Series Processing**: Support for daily, weekly, and monthly aggregation patterns
- **Feature Engineering**: Time-based features including moving averages, seasonality indicators, and trend components
- **Prediction Confidence**: Error metrics calculation using MAE and MSE for forecast accuracy assessment

### Modular Design Pattern
- **Utils Package**: Separated utility modules for data processing, visualizations, and forecasting
- **Page Components**: Isolated page rendering functions for maintainable code organization
- **Data Processor Class**: Centralized data handling with automatic column detection and cleaning methods
- **Forecaster Class**: Dedicated forecasting engine with configurable parameters and multiple aggregation levels

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the dashboard interface
- **Pandas**: Data manipulation and analysis library for Excel processing and data operations
- **NumPy**: Numerical computing support for calculations and array operations

### Visualization
- **Plotly Express**: High-level plotting interface for quick chart generation
- **Plotly Graph Objects**: Low-level plotting for custom interactive visualizations

### Machine Learning
- **Scikit-learn**: Machine learning library providing linear regression for forecasting
- **Linear Regression**: Primary forecasting algorithm for sales prediction
- **Metrics Module**: Model evaluation tools for forecast accuracy assessment

### Data Processing
- **Excel File Support**: Built-in pandas Excel reading capabilities for .xlsx and .xls files
- **DateTime Handling**: Python datetime module for time series operations
- **Regular Expressions**: Pattern matching for automatic column type detection

### Development Dependencies
- **Warnings Module**: Error suppression for cleaner user experience during model training
- **OS Module**: File system operations for potential file handling requirements

Note: The application uses a self-contained architecture with no external APIs or databases, processing uploaded files in memory for immediate analysis and visualization.