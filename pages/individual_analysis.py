import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.visualizations import create_salesperson_detail_chart, create_product_performance_chart

def render_individual_analysis(df, data_processor):
    """Render individual salesperson analysis page"""
    st.subheader("👤 Análisis Individual de Vendedores")
    
    if df.empty:
        st.warning("No hay datos disponibles para el análisis individual")
        return
    
    if not data_processor.salesperson_column:
        st.error("No se detectó una columna de vendedores en los datos")
        return
    
    # Salesperson selection
    salespeople = sorted(df[data_processor.salesperson_column].unique())
    selected_salesperson = st.selectbox(
        "Selecciona un vendedor para analizar:",
        salespeople,
        key="individual_salesperson_select"
    )
    
    if not selected_salesperson:
        st.warning("Por favor selecciona un vendedor")
        return
    
    # Filter data for selected salesperson
    salesperson_data = df[df[data_processor.salesperson_column] == selected_salesperson]
    
    if salesperson_data.empty:
        st.warning(f"No hay datos disponibles para {selected_salesperson}")
        return
    
    # Date filter
    col1, col2 = st.columns(2)
    with col1:
        try:
            min_date = salesperson_data[data_processor.date_column].min().date() if data_processor.date_column and not salesperson_data.empty else (datetime.now() - timedelta(days=365)).date()
        except:
            min_date = (datetime.now() - timedelta(days=365)).date()
            
        start_date = st.date_input(
            "Fecha Inicio",
            value=min_date,
            key="individual_start_date"
        )
    with col2:
        try:
            max_date = salesperson_data[data_processor.date_column].max().date() if data_processor.date_column and not salesperson_data.empty else datetime.now().date()
        except:
            max_date = datetime.now().date()
            
        end_date = st.date_input(
            "Fecha Fin",
            value=max_date,
            key="individual_end_date"
        )
    
    # Filter by date range
    if data_processor.date_column:
        filtered_data = data_processor.filter_by_date_range(salesperson_data, start_date, end_date)
    else:
        filtered_data = salesperson_data
    
    if filtered_data.empty:
        st.warning("No hay datos en el rango de fechas seleccionado")
        return
    
    # Individual KPIs
    st.subheader(f"📊 KPIs - {selected_salesperson}")
    
    if data_processor.amount_column:
        total_sales = filtered_data[data_processor.amount_column].sum()
        avg_sale = filtered_data[data_processor.amount_column].mean()
        num_transactions = len(filtered_data)
        max_sale = filtered_data[data_processor.amount_column].max()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Ventas Totales", f"${total_sales:,.2f}")
        
        with col2:
            st.metric("📊 Venta Promedio", f"${avg_sale:,.2f}")
        
        with col3:
            st.metric("🧾 Transacciones", f"{num_transactions:,}")
        
        with col4:
            st.metric("🎯 Venta Máxima", f"${max_sale:,.2f}")
        
        # Compare with team average
        if len(df) > len(filtered_data):  # Has team data
            team_avg_sale = df[data_processor.amount_column].mean()
            team_total_avg = df.groupby(data_processor.salesperson_column)[data_processor.amount_column].sum().mean()
            
            st.subheader("📈 Comparación con el Equipo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                performance_vs_avg = ((avg_sale - team_avg_sale) / team_avg_sale) * 100
                st.metric(
                    "📊 vs. Promedio del Equipo (Venta)",
                    f"${avg_sale:,.2f}",
                    delta=f"{performance_vs_avg:+.1f}%"
                )
            
            with col2:
                total_vs_avg = ((total_sales - team_total_avg) / team_total_avg) * 100
                st.metric(
                    "💰 vs. Promedio del Equipo (Total)",
                    f"${total_sales:,.2f}",
                    delta=f"{total_vs_avg:+.1f}%"
                )
    
    # Performance over time
    if data_processor.date_column and data_processor.amount_column:
        st.subheader("📈 Rendimiento en el Tiempo")
        fig_detail = create_salesperson_detail_chart(filtered_data, data_processor, selected_salesperson)
        st.plotly_chart(fig_detail, use_container_width=True)
        
        # Monthly performance table
        monthly_performance = filtered_data.groupby(
            filtered_data[data_processor.date_column].dt.to_period('M')
        )[data_processor.amount_column].agg(['sum', 'count', 'mean']).reset_index()
        
        monthly_performance.columns = ['Mes', 'Ventas Totales', 'Transacciones', 'Venta Promedio']
        monthly_performance['Mes'] = monthly_performance['Mes'].astype(str)
        monthly_performance['Ventas Totales'] = monthly_performance['Ventas Totales'].apply(lambda x: f"${x:,.2f}")
        monthly_performance['Venta Promedio'] = monthly_performance['Venta Promedio'].apply(lambda x: f"${x:,.2f}")
        
        st.subheader("📅 Rendimiento Mensual")
        st.dataframe(monthly_performance, use_container_width=True)
    
    # Product performance for the salesperson
    if data_processor.product_column and data_processor.amount_column:
        st.subheader("🛍️ Rendimiento por Producto")
        
        product_performance = filtered_data.groupby(data_processor.product_column)[data_processor.amount_column].agg([
            'sum', 'count', 'mean'
        ]).reset_index()
        product_performance.columns = ['Producto', 'Ventas Totales', 'Transacciones', 'Venta Promedio']
        product_performance = product_performance.sort_values('Ventas Totales', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top products chart
            top_products = product_performance.head(10)
            fig_products = px.bar(
                top_products,
                x='Ventas Totales',
                y='Producto',
                orientation='h',
                title=f"Top 10 Productos - {selected_salesperson}",
                color='Ventas Totales',
                color_continuous_scale='Blues'
            )
            fig_products.update_layout(template='plotly_white')
            st.plotly_chart(fig_products, use_container_width=True)
        
        with col2:
            # Product distribution pie chart
            fig_pie = px.pie(
                top_products,
                values='Ventas Totales',
                names='Producto',
                title="Distribución de Ventas por Producto"
            )
            fig_pie.update_layout(template='plotly_white')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Product performance table
        display_products = product_performance.copy()
        display_products['Ventas Totales'] = display_products['Ventas Totales'].apply(lambda x: f"${x:,.2f}")
        display_products['Venta Promedio'] = display_products['Venta Promedio'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(display_products, use_container_width=True)
    
    # Customer analysis
    if data_processor.customer_column and data_processor.amount_column:
        st.subheader("👥 Análisis de Clientes")
        
        customer_performance = filtered_data.groupby(data_processor.customer_column)[data_processor.amount_column].agg([
            'sum', 'count', 'mean'
        ]).reset_index()
        customer_performance.columns = ['Cliente', 'Ventas Totales', 'Transacciones', 'Venta Promedio']
        customer_performance = customer_performance.sort_values('Ventas Totales', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("👥 Total de Clientes", len(customer_performance))
            st.metric("🏆 Cliente Top", customer_performance.iloc[0]['Cliente'] if not customer_performance.empty else "N/A")
        
        with col2:
            avg_per_customer = customer_performance['Ventas Totales'].mean()
            st.metric("💰 Promedio por Cliente", f"${avg_per_customer:,.2f}")
            
            repeat_customers = len(customer_performance[customer_performance['Transacciones'] > 1])
            repeat_rate = (repeat_customers / len(customer_performance)) * 100
            st.metric("🔄 Tasa de Repetición", f"{repeat_rate:.1f}%")
        
        # Top customers chart
        top_customers = customer_performance.head(10)
        fig_customers = px.bar(
            top_customers,
            x='Cliente',
            y='Ventas Totales',
            title=f"Top 10 Clientes - {selected_salesperson}",
            color='Ventas Totales',
            color_continuous_scale='Greens'
        )
        fig_customers.update_xaxes(tickangle=45)
        fig_customers.update_layout(template='plotly_white')
        st.plotly_chart(fig_customers, use_container_width=True)
        
        # Customer performance table
        display_customers = top_customers.copy()
        display_customers['Ventas Totales'] = display_customers['Ventas Totales'].apply(lambda x: f"${x:,.2f}")
        display_customers['Venta Promedio'] = display_customers['Venta Promedio'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(display_customers, use_container_width=True)
    
    # Performance trends and insights
    if data_processor.date_column and data_processor.amount_column:
        st.subheader("📊 Tendencias y Insights")
        
        # Calculate growth rate
        monthly_data = filtered_data.groupby(
            filtered_data[data_processor.date_column].dt.to_period('M')
        )[data_processor.amount_column].sum()
        
        if len(monthly_data) >= 2:
            growth_rate = ((monthly_data.iloc[-1] - monthly_data.iloc[0]) / monthly_data.iloc[0]) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("📈 Crecimiento Total", f"{growth_rate:+.1f}%")
            
            with col2:
                avg_monthly_growth = monthly_data.pct_change().mean() * 100
                st.metric("📊 Crecimiento Promedio Mensual", f"{avg_monthly_growth:+.1f}%")
            
            # Seasonality analysis
            if len(monthly_data) >= 12:
                seasonal_data = filtered_data.copy()
                seasonal_data['month'] = seasonal_data[data_processor.date_column].dt.month
                monthly_avg = seasonal_data.groupby('month')[data_processor.amount_column].mean()
                
                fig_seasonal = px.line(
                    x=monthly_avg.index,
                    y=monthly_avg.values,
                    title="Patrón Estacional - Promedio de Ventas por Mes",
                    markers=True
                )
                fig_seasonal.update_layout(
                    xaxis_title="Mes",
                    yaxis_title="Venta Promedio ($)",
                    template='plotly_white',
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(1, 13)),
                        ticktext=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                                 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
                    )
                )
                st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # Export individual report
    st.subheader("📥 Exportar Reporte Individual")
    
    if st.button(f"Generar Reporte de {selected_salesperson}"):
        # Create summary report
        report_data = {
            'Vendedor': [selected_salesperson],
            'Ventas_Totales': [total_sales if data_processor.amount_column else 0],
            'Venta_Promedio': [avg_sale if data_processor.amount_column else 0],
            'Transacciones': [num_transactions if data_processor.amount_column else 0],
            'Venta_Maxima': [max_sale if data_processor.amount_column else 0]
        }
        
        if data_processor.product_column:
            report_data['Productos_Vendidos'] = [filtered_data[data_processor.product_column].nunique()]
        
        if data_processor.customer_column:
            report_data['Clientes_Atendidos'] = [filtered_data[data_processor.customer_column].nunique()]
        
        report_df = pd.DataFrame(report_data)
        csv = report_df.to_csv(index=False)
        
        st.download_button(
            label="📄 Descargar Reporte Individual",
            data=csv,
            file_name=f"reporte_{selected_salesperson.replace(' ', '_')}.csv",
            mime="text/csv"
        )
