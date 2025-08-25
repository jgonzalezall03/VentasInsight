import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SalesPredictionEngine:
    def __init__(self):
        self.sales_model = None
        self.performance_model = None
        self.label_encoders = {}
        
    def prepare_features(self, df, data_processor):
        """Prepare features for ML models"""
        if df.empty:
            return None
            
        features_df = df.copy()
        
        # Date features
        if data_processor.date_column:
            features_df['month'] = features_df[data_processor.date_column].dt.month
            features_df['quarter'] = features_df[data_processor.date_column].dt.quarter
            features_df['year'] = features_df[data_processor.date_column].dt.year
            features_df['day_of_week'] = features_df[data_processor.date_column].dt.dayofweek
        
        # Salesperson features
        if data_processor.salesperson_column:
            salesperson_stats = features_df.groupby(data_processor.salesperson_column)[data_processor.amount_column].agg(['mean', 'std', 'count']).reset_index()
            salesperson_stats.columns = [data_processor.salesperson_column, 'sp_avg_sale', 'sp_std_sale', 'sp_total_sales']
            features_df = features_df.merge(salesperson_stats, on=data_processor.salesperson_column, how='left')
        
        # Product features
        if data_processor.product_column:
            product_stats = features_df.groupby(data_processor.product_column)[data_processor.amount_column].agg(['mean', 'count']).reset_index()
            product_stats.columns = [data_processor.product_column, 'prod_avg_sale', 'prod_frequency']
            features_df = features_df.merge(product_stats, on=data_processor.product_column, how='left')
        
        return features_df
    
    def predict_next_month_sales(self, df, data_processor):
        """Predict sales for next month by salesperson"""
        try:
            features_df = self.prepare_features(df, data_processor)
            if features_df is None or len(features_df) < 10:
                return None
            
            # Aggregate by salesperson and month
            monthly_data = features_df.groupby([
                data_processor.salesperson_column,
                features_df[data_processor.date_column].dt.to_period('M')
            ])[data_processor.amount_column].sum().reset_index()
            
            monthly_data['month_num'] = monthly_data[data_processor.date_column].dt.month
            monthly_data['year'] = monthly_data[data_processor.date_column].dt.year
            
            # Create features for prediction
            X_features = []
            y_target = []
            
            for salesperson in monthly_data[data_processor.salesperson_column].unique():
                sp_data = monthly_data[monthly_data[data_processor.salesperson_column] == salesperson].sort_values(data_processor.date_column)
                
                if len(sp_data) >= 3:  # Need at least 3 months of data
                    for i in range(2, len(sp_data)):
                        # Features: last 2 months sales, month, trend
                        prev_2 = sp_data.iloc[i-2][data_processor.amount_column]
                        prev_1 = sp_data.iloc[i-1][data_processor.amount_column]
                        month = sp_data.iloc[i]['month_num']
                        trend = prev_1 - prev_2
                        
                        X_features.append([prev_2, prev_1, month, trend])
                        y_target.append(sp_data.iloc[i][data_processor.amount_column])
            
            if len(X_features) < 5:
                return None
            
            X = np.array(X_features)
            y = np.array(y_target)
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Predict next month for each salesperson
            predictions = []
            current_month = df[data_processor.date_column].max().month
            next_month = (current_month % 12) + 1
            
            for salesperson in monthly_data[data_processor.salesperson_column].unique():
                sp_data = monthly_data[monthly_data[data_processor.salesperson_column] == salesperson].sort_values(data_processor.date_column)
                
                if len(sp_data) >= 2:
                    last_2 = sp_data.iloc[-2][data_processor.amount_column]
                    last_1 = sp_data.iloc[-1][data_processor.amount_column]
                    trend = last_1 - last_2
                    
                    # Handle NaN and infinity values
                    if pd.isna(last_2) or np.isinf(last_2):
                        last_2 = 0
                    if pd.isna(last_1) or np.isinf(last_1):
                        last_1 = 0
                    if pd.isna(trend) or np.isinf(trend):
                        trend = 0
                    
                    # Clip extreme values
                    last_2 = np.clip(last_2, 0, 1000000)
                    last_1 = np.clip(last_1, 0, 1000000)
                    trend = np.clip(trend, -100000, 100000)
                    
                    pred_features = np.array([[last_2, last_1, next_month, trend]])
                    pred_features = np.nan_to_num(pred_features, nan=0, posinf=1000000, neginf=-1000000)
                    
                    prediction = model.predict(pred_features)[0]
                    prediction = max(0, np.clip(prediction, 0, 1000000))
                    
                    # Calculate confidence more robustly
                    if last_1 > 0:
                        confidence = min(100, max(50, 100 - abs(trend/last_1)*100))
                    else:
                        confidence = 50
                    
                    predictions.append({
                        'vendedor': salesperson,
                        'prediccion': prediction,
                        'ultimo_mes': last_1,
                        'confianza': confidence
                    })
            
            return sorted(predictions, key=lambda x: x['prediccion'], reverse=True)
            
        except Exception as e:
            st.error(f"Error en predicción: {str(e)}")
            return None
    
    def predict_top_performer_probability(self, df, data_processor):
        """Predict probability of each salesperson being top performer next month"""
        try:
            features_df = self.prepare_features(df, data_processor)
            if features_df is None or len(features_df) < 20:
                return None
            
            # Create monthly performance data
            monthly_perf = features_df.groupby([
                data_processor.salesperson_column,
                features_df[data_processor.date_column].dt.to_period('M')
            ])[data_processor.amount_column].sum().reset_index()
            
            # Determine top performer each month (top 20%)
            monthly_rankings = []
            for month in monthly_perf[data_processor.date_column].unique():
                month_data = monthly_perf[monthly_perf[data_processor.date_column] == month]
                threshold = month_data[data_processor.amount_column].quantile(0.8)
                month_data['is_top'] = (month_data[data_processor.amount_column] >= threshold).astype(int)
                monthly_rankings.append(month_data)
            
            ranking_df = pd.concat(monthly_rankings)
            
            # Create features for classification
            X_features = []
            y_target = []
            
            for salesperson in ranking_df[data_processor.salesperson_column].unique():
                sp_data = ranking_df[ranking_df[data_processor.salesperson_column] == salesperson].sort_values(data_processor.date_column)
                
                if len(sp_data) >= 3:
                    for i in range(2, len(sp_data)):
                        # Features: performance trend, consistency
                        sales_trend = sp_data.iloc[i-1:i+1][data_processor.amount_column].pct_change().iloc[-1]
                        avg_performance = sp_data.iloc[i-2:i][data_processor.amount_column].mean()
                        std_performance = sp_data.iloc[i-2:i][data_processor.amount_column].std()
                        
                        # Handle NaN and infinity values
                        if pd.isna(sales_trend) or np.isinf(sales_trend):
                            sales_trend = 0
                        if pd.isna(avg_performance) or np.isinf(avg_performance):
                            avg_performance = 0
                        if pd.isna(std_performance) or np.isinf(std_performance) or std_performance == 0:
                            consistency = 1
                        else:
                            consistency = 1 / (1 + std_performance)
                        
                        # Clip extreme values
                        sales_trend = np.clip(sales_trend, -10, 10)
                        avg_performance = np.clip(avg_performance, 0, 1000000)
                        consistency = np.clip(consistency, 0, 1)
                        
                        X_features.append([sales_trend, avg_performance, consistency])
                        y_target.append(sp_data.iloc[i]['is_top'])
            
            if len(X_features) < 10:
                return None
            
            X = np.array(X_features)
            y = np.array(y_target)
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0)
            
            # Train classifier
            model = GradientBoostingClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Predict for current salespeople
            current_predictions = []
            for salesperson in df[data_processor.salesperson_column].unique():
                sp_recent = df[df[data_processor.salesperson_column] == salesperson].tail(60)  # Last 2 months
                
                if len(sp_recent) >= 5:
                    recent_sales = sp_recent[data_processor.amount_column].sum()
                    sales_trend = sp_recent[data_processor.amount_column].pct_change().mean()
                    std_recent = sp_recent[data_processor.amount_column].std()
                    
                    # Handle NaN and infinity values
                    if pd.isna(sales_trend) or np.isinf(sales_trend):
                        sales_trend = 0
                    if pd.isna(recent_sales) or np.isinf(recent_sales):
                        recent_sales = 0
                    if pd.isna(std_recent) or np.isinf(std_recent) or std_recent == 0:
                        consistency = 1
                    else:
                        consistency = 1 / (1 + std_recent)
                    
                    # Clip extreme values
                    sales_trend = np.clip(sales_trend, -10, 10)
                    recent_sales = np.clip(recent_sales, 0, 1000000)
                    consistency = np.clip(consistency, 0, 1)
                    
                    pred_features = np.array([[sales_trend, recent_sales, consistency]])
                    pred_features = np.nan_to_num(pred_features, nan=0, posinf=1000000, neginf=-1000000)
                    
                    probability = model.predict_proba(pred_features)[0][1] * 100
                    probability = np.clip(probability, 0, 100)  # Ensure probability is between 0-100
                    
                    current_predictions.append({
                        'vendedor': salesperson,
                        'probabilidad': probability,
                        'ventas_recientes': recent_sales
                    })
            
            return sorted(current_predictions, key=lambda x: x['probabilidad'], reverse=True)
            
        except Exception as e:
            st.error(f"Error en predicción de top performer: {str(e)}")
            return None
    
    def predict_customer_churn_risk(self, df, data_processor):
        """Predict which customers are at risk of churning"""
        try:
            if not data_processor.customer_column or not data_processor.date_column:
                return None
            
            # Calculate customer metrics
            customer_metrics = df.groupby(data_processor.customer_column).agg({
                data_processor.amount_column: ['sum', 'mean', 'count'],
                data_processor.date_column: ['min', 'max']
            }).reset_index()
            
            customer_metrics.columns = ['customer', 'total_sales', 'avg_sale', 'frequency', 'first_purchase', 'last_purchase']
            
            # Calculate recency (days since last purchase)
            max_date = df[data_processor.date_column].max()
            customer_metrics['recency'] = (max_date - customer_metrics['last_purchase']).dt.days
            customer_metrics['customer_lifetime'] = (customer_metrics['last_purchase'] - customer_metrics['first_purchase']).dt.days
            
            # Simple churn risk scoring
            customer_metrics['churn_risk'] = 0
            
            # High recency = higher risk
            customer_metrics.loc[customer_metrics['recency'] > 90, 'churn_risk'] += 40
            customer_metrics.loc[customer_metrics['recency'] > 60, 'churn_risk'] += 30
            customer_metrics.loc[customer_metrics['recency'] > 30, 'churn_risk'] += 20
            
            # Low frequency = higher risk
            avg_frequency = customer_metrics['frequency'].mean()
            customer_metrics.loc[customer_metrics['frequency'] < avg_frequency * 0.5, 'churn_risk'] += 30
            
            # Declining sales trend
            for _, customer in customer_metrics.iterrows():
                customer_data = df[df[data_processor.customer_column] == customer['customer']].sort_values(data_processor.date_column)
                if len(customer_data) >= 3:
                    recent_avg = customer_data.tail(3)[data_processor.amount_column].mean()
                    older_avg = customer_data.head(len(customer_data)-3)[data_processor.amount_column].mean()
                    if recent_avg < older_avg * 0.8:
                        customer_metrics.loc[customer_metrics['customer'] == customer['customer'], 'churn_risk'] += 20
            
            # Cap at 100%
            customer_metrics['churn_risk'] = customer_metrics['churn_risk'].clip(0, 100)
            
            # Return top risk customers
            high_risk = customer_metrics[customer_metrics['churn_risk'] >= 50].sort_values('churn_risk', ascending=False)
            
            return high_risk[['customer', 'churn_risk', 'recency', 'total_sales', 'frequency']].head(10).to_dict('records')
            
        except Exception as e:
            st.error(f"Error en predicción de churn: {str(e)}")
            return None

def create_sales_prediction_chart(predictions):
    """Create chart for sales predictions"""
    if not predictions:
        return go.Figure().add_annotation(text="No hay datos suficientes para predicciones")
    
    df_pred = pd.DataFrame(predictions)
    
    fig = go.Figure()
    
    # Predicted sales
    fig.add_trace(go.Bar(
        x=df_pred['vendedor'],
        y=df_pred['prediccion'],
        name='Predicción Próximo Mes',
        marker_color='lightblue',
        text=df_pred['prediccion'].apply(lambda x: f'{x:,.0f} UF'.replace(',', '.')),
        textposition='outside'
    ))
    
    # Last month actual
    fig.add_trace(go.Bar(
        x=df_pred['vendedor'],
        y=df_pred['ultimo_mes'],
        name='Último Mes Real',
        marker_color='darkblue',
        text=df_pred['ultimo_mes'].apply(lambda x: f'{x:,.0f} UF'.replace(',', '.')),
        textposition='inside'
    ))
    
    fig.update_layout(
        title='🔮 Predicción de Ventas - Próximo Mes',
        xaxis_title='Vendedor',
        yaxis_title='Ventas (UF)',
        barmode='group',
        template='plotly_white'
    )
    
    return fig

def create_top_performer_probability_chart(predictions):
    """Create chart for top performer probabilities"""
    if not predictions:
        return go.Figure().add_annotation(text="No hay datos suficientes para predicciones")
    
    df_prob = pd.DataFrame(predictions)
    
    fig = px.bar(
        df_prob,
        x='vendedor',
        y='probabilidad',
        title='🎯 Probabilidad de ser Top Performer',
        color='probabilidad',
        color_continuous_scale='RdYlGn',
        text='probabilidad'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        xaxis_title='Vendedor',
        yaxis_title='Probabilidad (%)',
        template='plotly_white'
    )
    
    return fig

def create_churn_risk_chart(churn_data):
    """Create chart for customer churn risk"""
    if not churn_data:
        return go.Figure().add_annotation(text="No hay datos suficientes para análisis de churn")
    
    df_churn = pd.DataFrame(churn_data)
    
    fig = px.scatter(
        df_churn,
        x='recency',
        y='total_sales',
        size='churn_risk',
        color='churn_risk',
        hover_name='customer',
        title='⚠️ Riesgo de Pérdida de Clientes',
        color_continuous_scale='Reds',
        size_max=60
    )
    
    fig.update_layout(
        xaxis_title='Días desde Última Compra',
        yaxis_title='Ventas Totales (UF)',
        template='plotly_white'
    )
    
    return fig