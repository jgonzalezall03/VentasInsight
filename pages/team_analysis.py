import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.visualizations import create_team_performance_chart, create_monthly_comparison_chart, create_regional_performance_chart

def render_team_analysis(df, data_processor):
    """Render team analysis page"""
    st.subheader("👥 Análisis del Equipo de Ventas")
    
    if df.empty:
        st.warning("No hay datos disponibles para el análisis del equipo")
        return
    
    # Date filter
    col1, col2 = st.columns(2)
    with col1:
        try:
            min_date = df[data_processor.date_column].min().date() if data_processor.date_column and not df.empty else (datetime.now() - timedelta(days=365)).date()
        except:
            min_date = (datetime.now() - timedelta(days=365)).date()
            
        start_date = st.date_input(
            "Fecha Inicio",
            value=min_date,
            key="team_start_date"
        )
    with col2:
        try:
            max_date = df[data_processor.date_column].max().date() if data_processor.date_column and not df.empty else datetime.now().date()
        except:
            max_date = datetime.now().date()
            
        end_date = st.date_input(
            "Fecha Fin",
            value=max_date,
            key="team_end_date"
        )
    
    # Filter data
    if data_processor.date_column:
        filtered_df = data_processor.filter_by_date_range(df, start_date, end_date)
    else:
        filtered_df = df
    
    if filtered_df.empty:
        st.warning("No hay datos en el rango de fechas seleccionado")
        return
    
    # Team Overview
    st.subheader("📊 Resumen del Equipo")
    
    if data_processor.salesperson_column and data_processor.amount_column:
        # Team performance metrics
        team_metrics = filtered_df.groupby(data_processor.salesperson_column)[data_processor.amount_column].agg([
            'sum', 'count', 'mean', 'std'
        ]).reset_index()
        team_metrics.columns = ['vendedor', 'ventas_totales', 'transacciones', 'venta_promedio', 'desviacion']
        team_metrics = team_metrics.sort_values('ventas_totales', ascending=False)
        
        # Top performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("🏆 **Top 3 Vendedores por Ventas**")
            top_3 = team_metrics.head(3)
            for i, row in top_3.iterrows():
                st.write(f"{i+1}. **{row['vendedor']}** - ${row['ventas_totales']:,.2f}")
        
        with col2:
            st.markdown("📈 **Top 3 Vendedores por Transacciones**")
            top_3_trans = team_metrics.sort_values('transacciones', ascending=False).head(3)
            for i, row in top_3_trans.iterrows():
                st.write(f"{i+1}. **{row['vendedor']}** - {row['transacciones']} transacciones")
        
        # Team performance chart
        fig_team = create_team_performance_chart(filtered_df, data_processor)
        st.plotly_chart(fig_team, use_container_width=True, key="team_performance_main")
        
        # Detailed team statistics table
        st.subheader("📋 Estadísticas Detalladas del Equipo")
        
        # Format the metrics table
        display_metrics = team_metrics.copy()
        display_metrics['ventas_totales'] = display_metrics['ventas_totales'].apply(lambda x: f"${x:,.2f}")
        display_metrics['venta_promedio'] = display_metrics['venta_promedio'].apply(lambda x: f"${x:,.2f}")
        display_metrics['desviacion'] = display_metrics['desviacion'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
        
        display_metrics.columns = ['Vendedor', 'Ventas Totales', 'Transacciones', 'Venta Promedio', 'Desviación Estándar']
        st.dataframe(display_metrics, use_container_width=True)
        
        # Performance distribution
        st.subheader("📊 Distribución del Rendimiento")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales distribution
            fig_dist = px.box(
                filtered_df,
                x=data_processor.salesperson_column,
                y=data_processor.amount_column,
                title="Distribución de Ventas por Vendedor"
            )
            fig_dist.update_xaxes(tickangle=45)
            fig_dist.update_layout(template='plotly_white')
            st.plotly_chart(fig_dist, use_container_width=True, key="team_sales_distribution")
        
        with col2:
            # Transaction count by salesperson
            transaction_counts = filtered_df[data_processor.salesperson_column].value_counts()
            fig_trans = px.pie(
                values=transaction_counts.values,
                names=transaction_counts.index,
                title="Distribución de Transacciones"
            )
            fig_trans.update_layout(template='plotly_white')
            st.plotly_chart(fig_trans, use_container_width=True, key="team_transactions_pie")
    
    # Monthly comparison if date data available
    if data_processor.date_column:
        st.subheader("📅 Análisis Temporal")
        fig_monthly = create_monthly_comparison_chart(filtered_df, data_processor)
        st.plotly_chart(fig_monthly, use_container_width=True, key="team_monthly_comparison")
        
        # Monthly team performance
        if data_processor.salesperson_column:
            st.subheader("📊 Rendimiento Mensual por Vendedor")
            
            # Create month-year column
            filtered_df_copy = filtered_df.copy()
            filtered_df_copy['month_year'] = filtered_df_copy[data_processor.date_column].dt.to_period('M').astype(str)
            
            # Aggregate by month and salesperson
            monthly_team = filtered_df_copy.groupby(['month_year', data_processor.salesperson_column])[
                data_processor.amount_column
            ].sum().reset_index()
            
            # Pivot for better visualization
            monthly_pivot = monthly_team.pivot(
                index='month_year',
                columns=data_processor.salesperson_column,
                values=data_processor.amount_column
            ).fillna(0)
            
            # Create heatmap
            fig_heatmap = px.imshow(
                monthly_pivot.T,
                title="Mapa de Calor - Ventas Mensuales por Vendedor",
                color_continuous_scale="Blues",
                aspect="auto"
            )
            fig_heatmap.update_layout(
                xaxis_title="Mes",
                yaxis_title="Vendedor",
                template='plotly_white'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True, key="team_heatmap")
    
    # Regional analysis if available
    if data_processor.region_column:
        st.subheader("🌍 Análisis Regional")
        fig_regional = create_regional_performance_chart(filtered_df, data_processor)
        st.plotly_chart(fig_regional, use_container_width=True, key="team_regional_chart")
        
        # Regional team performance
        if data_processor.salesperson_column:
            regional_team = filtered_df.groupby([data_processor.region_column, data_processor.salesperson_column])[
                data_processor.amount_column
            ].sum().reset_index()
            
            fig_regional_team = px.treemap(
                regional_team,
                path=[data_processor.region_column, data_processor.salesperson_column],
                values=data_processor.amount_column,
                title="Treemap - Ventas por Región y Vendedor"
            )
            fig_regional_team.update_layout(template='plotly_white')
            st.plotly_chart(fig_regional_team, use_container_width=True, key="team_regional_treemap")
    
    # Team collaboration metrics
    if data_processor.customer_column and data_processor.salesperson_column:
        st.subheader("🤝 Métricas de Colaboración")
        
        # Customer overlap analysis
        customer_sales = filtered_df.groupby(data_processor.customer_column)[data_processor.salesperson_column].nunique()
        shared_customers = customer_sales[customer_sales > 1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "👥 Clientes Compartidos",
                len(shared_customers),
                delta=f"{len(shared_customers)/len(customer_sales)*100:.1f}% del total"
            )
        
        with col2:
            avg_salespeople_per_customer = customer_sales.mean()
            st.metric(
                "📊 Promedio Vendedores/Cliente",
                f"{avg_salespeople_per_customer:.2f}",
                delta=None
            )
    
    # Export functionality
    st.subheader("📥 Exportar Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Descargar Reporte del Equipo"):
            if data_processor.salesperson_column and data_processor.amount_column:
                csv = team_metrics.to_csv(index=False)
                st.download_button(
                    label="📄 Descargar CSV",
                    data=csv,
                    file_name="reporte_equipo_ventas.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("Descargar Datos Filtrados"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📄 Descargar Datos CSV",
                data=csv,
                file_name="datos_ventas_filtrados.csv",
                mime="text/csv"
            )
