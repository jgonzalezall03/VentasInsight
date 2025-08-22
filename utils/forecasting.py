import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class SalesForecaster:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.model = None
        self.historical_data = None
        
    def prepare_forecasting_data(self, df, aggregation='monthly'):
        """Prepare data for forecasting"""
        if not self.data_processor.date_column or not self.data_processor.amount_column:
            raise ValueError("Date and amount columns are required for forecasting")
        
        # Aggregate data by time period
        if aggregation == 'monthly':
            df_agg = df.groupby(df[self.data_processor.date_column].dt.to_period('M'))[
                self.data_processor.amount_column
            ].sum().reset_index()
            df_agg[self.data_processor.date_column] = df_agg[self.data_processor.date_column].dt.to_timestamp()
        elif aggregation == 'weekly':
            df_agg = df.groupby(df[self.data_processor.date_column].dt.to_period('W'))[
                self.data_processor.amount_column
            ].sum().reset_index()
            df_agg[self.data_processor.date_column] = df_agg[self.data_processor.date_column].dt.to_timestamp()
        else:  # daily
            df_agg = df.groupby(df[self.data_processor.date_column].dt.date)[
                self.data_processor.amount_column
            ].sum().reset_index()
            df_agg[self.data_processor.date_column] = pd.to_datetime(df_agg[self.data_processor.date_column])
        
        # Sort by date
        df_agg = df_agg.sort_values(self.data_processor.date_column)
        
        # Create time-based features
        df_agg['time_index'] = range(len(df_agg))
        df_agg['month'] = df_agg[self.data_processor.date_column].dt.month
        df_agg['quarter'] = df_agg[self.data_processor.date_column].dt.quarter
        df_agg['year'] = df_agg[self.data_processor.date_column].dt.year
        
        # Add moving averages
        df_agg['ma_3'] = df_agg[self.data_processor.amount_column].rolling(window=3, min_periods=1).mean()
        df_agg['ma_6'] = df_agg[self.data_processor.amount_column].rolling(window=6, min_periods=1).mean()
        
        # Add trend
        if len(df_agg) > 1:
            df_agg['trend'] = np.polyval(np.polyfit(df_agg['time_index'], df_agg[self.data_processor.amount_column], 1), df_agg['time_index'])
        else:
            df_agg['trend'] = df_agg[self.data_processor.amount_column]
        
        self.historical_data = df_agg
        return df_agg
    
    def create_linear_regression_forecast(self, df_agg, periods=6):
        """Create forecast using linear regression"""
        # Prepare features
        features = ['time_index', 'month', 'quarter']
        X = df_agg[features]
        y = df_agg[self.data_processor.amount_column]
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        # Create future dates
        last_date = df_agg[self.data_processor.date_column].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        # Create future features
        future_df = pd.DataFrame({
            self.data_processor.date_column: future_dates,
            'time_index': range(len(df_agg), len(df_agg) + periods),
            'month': future_dates.month,
            'quarter': future_dates.quarter
        })
        
        # Generate predictions
        future_X = future_df[features]
        predictions = self.model.predict(future_X)
        
        # Add confidence intervals (simple approach)
        historical_residuals = y - self.model.predict(X)
        std_residual = np.std(historical_residuals)
        
        future_df['forecast'] = predictions
        future_df['lower_bound'] = predictions - 1.96 * std_residual
        future_df['upper_bound'] = predictions + 1.96 * std_residual
        
        # Ensure non-negative values
        future_df['forecast'] = np.maximum(future_df['forecast'], 0)
        future_df['lower_bound'] = np.maximum(future_df['lower_bound'], 0)
        
        return future_df
    
    def create_moving_average_forecast(self, df_agg, periods=6, window=3):
        """Create forecast using moving average"""
        last_values = df_agg[self.data_processor.amount_column].tail(window).values
        avg_value = np.mean(last_values)
        
        # Calculate trend
        if len(df_agg) >= 6:
            recent_trend = np.polyfit(range(window), last_values, 1)[0]
        else:
            recent_trend = 0
        
        # Create future dates
        last_date = df_agg[self.data_processor.date_column].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        # Generate predictions with trend
        predictions = []
        for i in range(periods):
            pred = avg_value + (recent_trend * (i + 1))
            predictions.append(max(pred, 0))  # Ensure non-negative
        
        # Calculate confidence intervals
        std_historical = np.std(df_agg[self.data_processor.amount_column])
        
        future_df = pd.DataFrame({
            self.data_processor.date_column: future_dates,
            'forecast': predictions,
            'lower_bound': np.maximum(np.array(predictions) - 1.96 * std_historical, 0),
            'upper_bound': np.array(predictions) + 1.96 * std_historical
        })
        
        return future_df
    
    def create_seasonal_forecast(self, df_agg, periods=6):
        """Create forecast considering seasonality"""
        # Calculate monthly averages for seasonality
        df_agg['month'] = df_agg[self.data_processor.date_column].dt.month
        monthly_avg = df_agg.groupby('month')[self.data_processor.amount_column].mean()
        
        # Calculate overall trend
        if len(df_agg) > 1:
            overall_trend = np.polyfit(df_agg['time_index'], df_agg[self.data_processor.amount_column], 1)[0]
        else:
            overall_trend = 0
        
        # Create future dates
        last_date = df_agg[self.data_processor.date_column].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
        
        # Generate seasonal predictions
        predictions = []
        for i, date in enumerate(future_dates):
            month = date.month
            seasonal_factor = monthly_avg.get(month, df_agg[self.data_processor.amount_column].mean())
            trend_component = overall_trend * (len(df_agg) + i + 1)
            pred = seasonal_factor + trend_component
            predictions.append(max(pred, 0))
        
        # Calculate confidence intervals
        std_historical = np.std(df_agg[self.data_processor.amount_column])
        
        future_df = pd.DataFrame({
            self.data_processor.date_column: future_dates,
            'forecast': predictions,
            'lower_bound': np.maximum(np.array(predictions) - 1.96 * std_historical, 0),
            'upper_bound': np.array(predictions) + 1.96 * std_historical
        })
        
        return future_df
    
    def evaluate_model_accuracy(self, df_agg, test_size=0.2):
        """Evaluate model accuracy using train-test split"""
        if len(df_agg) < 6:
            return {"error": "Insufficient data for model evaluation"}
        
        # Split data
        split_idx = int(len(df_agg) * (1 - test_size))
        train_data = df_agg[:split_idx]
        test_data = df_agg[split_idx:]
        
        # Train model on training data
        features = ['time_index', 'month', 'quarter']
        X_train = train_data[features]
        y_train = train_data[self.data_processor.amount_column]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict on test data
        X_test = test_data[features]
        y_test = test_data[self.data_processor.amount_column]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': model.score(X_test, y_test)
        }
    
    def create_forecast_chart(self, df_agg, forecast_df, title="Sales Forecast"):
        """Create interactive forecast chart"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df_agg[self.data_processor.date_column],
            y=df_agg[self.data_processor.amount_column],
            mode='lines+markers',
            name='Datos Históricos',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df[self.data_processor.date_column],
            y=forecast_df['forecast'],
            mode='lines+markers',
            name='Pronóstico',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df[self.data_processor.date_column],
            y=forecast_df['upper_bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df[self.data_processor.date_column],
            y=forecast_df['lower_bound'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            name='Intervalo de Confianza',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Fecha',
            yaxis_title='Ventas (UF)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
