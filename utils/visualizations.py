import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st

def create_kpi_cards(df, data_processor):
    """Create KPI cards for the dashboard"""
    if df.empty or not data_processor.amount_column:
        st.warning("No hay datos suficientes para mostrar KPIs")
        return
    
    # Calculate KPIs
    total_sales = df[data_processor.amount_column].sum()
    avg_sale = df[data_processor.amount_column].mean()
    
    # Count contracts - use contract column from Excel if available
    if data_processor.contract_column:
        # Use the contract column directly from Excel
        num_contracts = df[data_processor.contract_column].sum()
    elif data_processor.customer_column and data_processor.date_column:
        # Count unique contracts as unique customer-month combinations
        df_contracts = df.copy()
        df_contracts['month_year'] = df_contracts[data_processor.date_column].dt.to_period('M')
        num_contracts = df_contracts.groupby([data_processor.customer_column, 'month_year']).size().count()
    else:
        # Fallback to counting total transactions
        num_contracts = len(df)
    
    # Additional KPIs if columns available
    num_salespeople = df[data_processor.salesperson_column].nunique() if data_processor.salesperson_column else 0
    
    # Create columns for KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Ventas Totales",
            value=f"{total_sales:,.2f} UF".replace(',', 'X').replace('.', ',').replace('X', '.'),
            delta=None
        )
    
    with col2:
        st.metric(
            label="📊 Venta Promedio",
            value=f"{avg_sale:,.2f} UF".replace(',', 'X').replace('.', ',').replace('X', '.'),
            delta=None
        )
    
    with col3:
        st.metric(
            label="🧾 Contratos",
            value=f"{num_contracts:,}",
            delta=None
        )
    
    with col4:
        if num_salespeople > 0:
            st.metric(
                label="👥 Vendedores",
                value=f"{num_salespeople}",
                delta=None
            )
        else:
            avg_daily = total_sales / max(1, (df[data_processor.date_column].max() - df[data_processor.date_column].min()).days) if data_processor.date_column else 0
            st.metric(
                label="📈 Promedio Diario",
                value=f"{avg_daily:,.2f} UF".replace(',', 'X').replace('.', ',').replace('X', '.'),
                delta=None
            )

def create_sales_trend_chart(df, data_processor):
    """Create sales trend chart over time"""
    if df.empty or not data_processor.date_column or not data_processor.amount_column:
        return go.Figure().add_annotation(text="No hay datos suficientes para el gráfico de tendencias")
    
    # Aggregate by month
    monthly_sales = df.groupby(df[data_processor.date_column].dt.to_period('M'))[data_processor.amount_column].sum().reset_index()
    monthly_sales[data_processor.date_column] = monthly_sales[data_processor.date_column].dt.to_timestamp()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_sales[data_processor.date_column],
        y=monthly_sales[data_processor.amount_column],
        mode='lines+markers',
        name='Ventas Mensuales',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Add trend line
    if len(monthly_sales) > 1:
        z = np.polyfit(range(len(monthly_sales)), monthly_sales[data_processor.amount_column], 1)
        trend_line = np.poly1d(z)(range(len(monthly_sales)))
        
        fig.add_trace(go.Scatter(
            x=monthly_sales[data_processor.date_column],
            y=trend_line,
            mode='lines',
            name='Tendencia',
            line=dict(color='red', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title='📈 Tendencia de Ventas Mensuales',
        xaxis_title='Fecha',
        yaxis_title='Ventas (UF)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def create_team_performance_chart(df, data_processor):
    """Create team performance comparison chart"""
    if df.empty or not data_processor.salesperson_column or not data_processor.amount_column:
        return go.Figure().add_annotation(text="No hay datos suficientes para el análisis del equipo")
    
    # Aggregate by salesperson
    performance = df.groupby(data_processor.salesperson_column)[data_processor.amount_column].agg([
        'sum', 'count', 'mean'
    ]).reset_index()
    
    performance.columns = ['salesperson', 'total_sales', 'transactions', 'avg_sale']
    performance = performance.sort_values('total_sales', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=performance['salesperson'],
        x=performance['total_sales'],
        orientation='h',
        name='Ventas Totales',
        marker_color='#2E86AB',
        text=performance['total_sales'].apply(lambda x: f'{x:,.0f} UF'),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='👥 Rendimiento del Equipo de Ventas',
        xaxis_title='Ventas Totales (UF)',
        yaxis_title='Vendedor',
        template='plotly_white',
        height=max(400, len(performance) * 50)
    )
    
    return fig

def create_product_performance_chart(df, data_processor):
    """Create product performance chart"""
    if df.empty or not data_processor.product_column or not data_processor.amount_column:
        return go.Figure().add_annotation(text="No hay datos de productos disponibles")
    
    # Aggregate by product
    product_sales = df.groupby(data_processor.product_column)[data_processor.amount_column].sum().reset_index()
    product_sales = product_sales.sort_values(data_processor.amount_column, ascending=False).head(10)
    
    fig = px.pie(
        product_sales,
        values=data_processor.amount_column,
        names=data_processor.product_column,
        title='🛍️ Top 10 Productos por Ventas'
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template='plotly_white')
    
    return fig

def create_monthly_comparison_chart(df, data_processor):
    """Create month-over-month comparison chart"""
    if df.empty or not data_processor.date_column or not data_processor.amount_column:
        return go.Figure().add_annotation(text="No hay datos suficientes para comparación mensual")
    
    # Create month and year columns
    df_copy = df.copy()
    df_copy['month'] = df_copy[data_processor.date_column].dt.month
    df_copy['year'] = df_copy[data_processor.date_column].dt.year
    
    # Aggregate by month and year
    monthly_data = df_copy.groupby(['year', 'month'])[data_processor.amount_column].sum().reset_index()
    
    fig = go.Figure()
    
    # Group by year
    for year in monthly_data['year'].unique():
        year_data = monthly_data[monthly_data['year'] == year]
        
        fig.add_trace(go.Scatter(
            x=year_data['month'],
            y=year_data[data_processor.amount_column],
            mode='lines+markers',
            name=f'Año {year}',
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='📅 Comparación de Ventas por Mes',
        xaxis_title='Mes',
        yaxis_title='Ventas (UF)',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        ),
        template='plotly_white'
    )
    
    return fig

def create_salesperson_detail_chart(df, data_processor, salesperson):
    """Create detailed chart for specific salesperson"""
    if df.empty or not data_processor.salesperson_column:
        return go.Figure().add_annotation(text="No hay datos del vendedor disponibles")
    
    # Filter data for specific salesperson
    salesperson_data = df[df[data_processor.salesperson_column] == salesperson]
    
    if salesperson_data.empty:
        return go.Figure().add_annotation(text=f"No hay datos para {salesperson}")
    
    # Create monthly aggregation
    if data_processor.contract_column:
        # Use contract column from Excel
        monthly_sales = salesperson_data.groupby(
            salesperson_data[data_processor.date_column].dt.to_period('M')
        ).agg({
            data_processor.amount_column: 'sum',
            data_processor.contract_column: 'sum'
        }).reset_index()
        monthly_sales.columns = [data_processor.date_column, 'sum', 'count']
    else:
        # Fallback to counting transactions
        monthly_sales = salesperson_data.groupby(
            salesperson_data[data_processor.date_column].dt.to_period('M')
        )[data_processor.amount_column].agg(['sum', 'count']).reset_index()
    
    monthly_sales[data_processor.date_column] = monthly_sales[data_processor.date_column].dt.to_timestamp()
    
    # Create subplot
    fig = go.Figure()
    
    # Sales amount
    fig.add_trace(go.Scatter(
        x=monthly_sales[data_processor.date_column],
        y=monthly_sales['sum'],
        mode='lines+markers',
        name='Ventas (UF)',
        yaxis='y',
        line=dict(color='blue', width=3)
    ))
    
    # Number of contracts
    fig.add_trace(go.Scatter(
        x=monthly_sales[data_processor.date_column],
        y=monthly_sales['count'],
        mode='lines+markers',
        name='# Contratos',
        yaxis='y2',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title=f'📊 Rendimiento Detallado - {salesperson}',
        xaxis_title='Fecha',
        yaxis=dict(
            title='Ventas (UF)',
            side='left'
        ),
        yaxis2=dict(
            title='Número de Contratos',
            side='right',
            overlaying='y'
        ),
        template='plotly_white'
    )
    
    return fig

def create_regional_performance_chart(df, data_processor):
    """Create regional performance chart"""
    if df.empty or not data_processor.region_column or not data_processor.amount_column:
        return go.Figure().add_annotation(text="No hay datos de región disponibles")
    
    # Aggregate by region
    regional_sales = df.groupby(data_processor.region_column)[data_processor.amount_column].sum().reset_index()
    regional_sales = regional_sales.sort_values(data_processor.amount_column, ascending=False)
    
    fig = px.bar(
        regional_sales,
        x=data_processor.region_column,
        y=data_processor.amount_column,
        title='🌍 Ventas por Región',
        color=data_processor.amount_column,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title='Región',
        yaxis_title='Ventas (UF)',
        template='plotly_white'
    )
    
    return fig

def create_performance_matrix(df, data_processor):
    """Create performance matrix chart (Sales vs Contracts)"""
    if df.empty or not data_processor.salesperson_column or not data_processor.amount_column:
        return go.Figure().add_annotation(text="No hay datos suficientes para la matriz de rendimiento")
    
    # Calculate metrics by salesperson
    if data_processor.contract_column:
        metrics = df.groupby(data_processor.salesperson_column).agg({
            data_processor.amount_column: 'sum',
            data_processor.contract_column: 'sum'
        }).reset_index()
        metrics.columns = ['vendedor', 'ventas_totales', 'contratos_totales']
    else:
        metrics = df.groupby(data_processor.salesperson_column)[data_processor.amount_column].agg(['sum', 'count']).reset_index()
        metrics.columns = ['vendedor', 'ventas_totales', 'contratos_totales']
    
    # Calculate medians for quadrant division
    median_sales = metrics['ventas_totales'].median()
    median_contracts = metrics['contratos_totales'].median()
    
    # Classify into quadrants
    def classify_quadrant(row):
        if row['ventas_totales'] >= median_sales and row['contratos_totales'] >= median_contracts:
            return 'Estrellas'
        elif row['ventas_totales'] >= median_sales and row['contratos_totales'] < median_contracts:
            return 'Vacas Lecheras'
        elif row['ventas_totales'] < median_sales and row['contratos_totales'] >= median_contracts:
            return 'Interrogantes'
        else:
            return 'Perros'
    
    metrics['cuadrante'] = metrics.apply(classify_quadrant, axis=1)
    
    # Remove rows with NaN values and ensure positive values for size
    metrics = metrics.dropna(subset=['ventas_totales', 'contratos_totales'])
    metrics = metrics[metrics['ventas_totales'] > 0]
    
    # Create scatter plot
    fig = px.scatter(
        metrics,
        x='contratos_totales',
        y='ventas_totales',
        color='cuadrante',
        size='ventas_totales',
        hover_name='vendedor',
        title='Matriz de Rendimiento: Ventas vs Contratos',
        color_discrete_map={
            'Estrellas': '#2E8B57',
            'Vacas Lecheras': '#4169E1', 
            'Interrogantes': '#FFD700',
            'Perros': '#DC143C'
        }
    )
    
    # Add quadrant lines
    fig.add_hline(y=median_sales, line_dash="dash", line_color="gray")
    fig.add_vline(x=median_contracts, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        xaxis_title='Contratos Totales',
        yaxis_title='Ventas Totales (UF)',
        template='plotly_white'
    )
    
    return fig

def create_temporal_heatmap(df, data_processor):
    """Create temporal heatmap (Salesperson vs Month)"""
    if df.empty or not data_processor.salesperson_column or not data_processor.date_column or not data_processor.amount_column:
        return go.Figure().add_annotation(text="No hay datos suficientes para el heatmap temporal")
    
    # Create month-year column
    df_copy = df.copy()
    df_copy['month_year'] = df_copy[data_processor.date_column].dt.to_period('M').astype(str)
    
    # Aggregate by month and salesperson
    heatmap_data = df_copy.groupby(['month_year', data_processor.salesperson_column])[data_processor.amount_column].sum().reset_index()
    
    # Pivot for heatmap
    heatmap_pivot = heatmap_data.pivot(
        index=data_processor.salesperson_column,
        columns='month_year',
        values=data_processor.amount_column
    ).fillna(0)
    
    # Create heatmap
    fig = px.imshow(
        heatmap_pivot,
        title='Heatmap Temporal: Ventas por Vendedor y Mes',
        color_continuous_scale='RdYlBu_r',
        aspect='auto'
    )
    
    fig.update_layout(
        xaxis_title='Mes',
        yaxis_title='Vendedor',
        template='plotly_white'
    )
    
    return fig

def create_efficiency_analysis(df, data_processor):
    """Create efficiency analysis chart (Sales per Contract)"""
    if df.empty or not data_processor.salesperson_column or not data_processor.amount_column:
        return go.Figure().add_annotation(text="No hay datos suficientes para el análisis de eficiencia")
    
    # Calculate efficiency metrics
    if data_processor.contract_column:
        efficiency_data = df.groupby(data_processor.salesperson_column).agg({
            data_processor.amount_column: 'sum',
            data_processor.contract_column: 'sum'
        }).reset_index()
        efficiency_data.columns = ['vendedor', 'ventas_totales', 'contratos_totales']
    else:
        efficiency_data = df.groupby(data_processor.salesperson_column)[data_processor.amount_column].agg(['sum', 'count']).reset_index()
        efficiency_data.columns = ['vendedor', 'ventas_totales', 'contratos_totales']
    
    # Calculate efficiency (sales per contract) and handle NaN/inf values
    efficiency_data['eficiencia'] = efficiency_data['ventas_totales'] / efficiency_data['contratos_totales']
    efficiency_data['eficiencia'] = efficiency_data['eficiencia'].replace([np.inf, -np.inf], np.nan)
    efficiency_data = efficiency_data.dropna(subset=['eficiencia'])
    efficiency_data = efficiency_data.sort_values('eficiencia', ascending=False)
    
    # Ensure positive values for bubble sizes
    efficiency_data = efficiency_data[efficiency_data['eficiencia'] > 0]
    
    # Create bubble chart
    fig = px.scatter(
        efficiency_data,
        x='contratos_totales',
        y='ventas_totales',
        size='eficiencia',
        color='eficiencia',
        hover_name='vendedor',
        title='Análisis de Eficiencia: Venta por Contrato',
        color_continuous_scale='Viridis',
        size_max=60
    )
    
    fig.update_layout(
        xaxis_title='Contratos Totales',
        yaxis_title='Ventas Totales (UF)',
        template='plotly_white'
    )
    
    return fig
