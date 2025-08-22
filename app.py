import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from utils.data_processor import DataProcessor
from utils.visualizations import create_kpi_cards, create_team_performance_chart, create_sales_trend_chart
import os

# Configure page
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'df' not in st.session_state:
    st.session_state.df = None

def main():
    st.title("📊 Sales Analytics Dashboard")
    st.markdown("### Análisis de Rendimiento del Equipo de Ventas y Forecasting")
    
    # Sidebar for file upload and navigation
    with st.sidebar:
        st.header("📁 Carga de Datos")
        
        uploaded_file = st.file_uploader(
            "Sube tu archivo Excel de ventas",
            type=['xlsx', 'xls'],
            help="Formatos soportados: .xlsx, .xls"
        )
        
        # Botón para cargar archivo de ejemplo
        if st.button("📊 Cargar archivo de ejemplo del proyecto"):
            try:
                import os
                if os.path.exists("attached_assets/ventas_limpias_1755883892344.xlsx"):
                    with st.spinner("Cargando archivo de ejemplo..."):
                        st.session_state.data_processor = DataProcessor()
                        st.session_state.df = st.session_state.data_processor.load_and_process_excel("attached_assets/ventas_limpias_1755883892344.xlsx")
                    st.success("✅ Archivo de ejemplo cargado exitosamente!")
                    st.rerun()
                else:
                    st.error("No se encontró el archivo de ejemplo")
            except Exception as e:
                st.error(f"Error al cargar el archivo de ejemplo: {str(e)}")
        
        if uploaded_file is not None:
            try:
                with st.spinner("Procesando archivo..."):
                    # Initialize data processor
                    st.session_state.data_processor = DataProcessor()
                    st.session_state.df = st.session_state.data_processor.load_and_process_excel(uploaded_file)
                
                st.success(f"✅ Archivo cargado exitosamente!")
                st.info(f"📋 {len(st.session_state.df)} registros encontrados")
                
                # Show data info
                if st.session_state.df is not None:
                    st.subheader("🔍 Información del Dataset")
                    st.write(f"**Columnas detectadas:** {len(st.session_state.df.columns)}")
                    
                    # Show detected columns
                    detected_cols = st.session_state.data_processor.get_detected_columns()
                    if detected_cols:
                        st.write("**Columnas identificadas:**")
                        for col_type, col_name in detected_cols.items():
                            if col_name:
                                st.write(f"- {col_type.title()}: `{col_name}`")
                    
            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {str(e)}")
                st.info("💡 Asegúrate de que el archivo contenga columnas de ventas típicas como: fecha, vendedor, monto, producto, etc.")
    
    # Main content area
    if st.session_state.df is not None and st.session_state.data_processor is not None:
        # Navigation tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏠 Dashboard General", 
            "👥 Análisis por Equipo", 
            "👤 Análisis Individual", 
            "📈 Forecasting"
        ])
        
        with tab1:
            show_general_dashboard()
        
        with tab2:
            show_team_analysis()
        
        with tab3:
            show_individual_analysis()
        
        with tab4:
            show_forecasting()
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>🎯 Bienvenido al Dashboard de Ventas</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Sube tu archivo Excel de ventas para comenzar el análisis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("""
            **📋 Funcionalidades disponibles:**
            
            ✅ **Dashboard General**: KPIs principales y métricas de rendimiento
            
            ✅ **Análisis por Equipo**: Comparación de rendimiento entre vendedores
            
            ✅ **Análisis Individual**: Métricas detalladas por ejecutivo
            
            ✅ **Forecasting**: Predicciones de ventas para los próximos meses
            
            ✅ **Filtros Interactivos**: Por fecha, vendedor, producto y más
            """)

def show_general_dashboard():
    """Display general dashboard with KPIs and overview charts"""
    st.subheader("📊 Dashboard General")
    
    df = st.session_state.df
    processor = st.session_state.data_processor
    
    # Date filter
    col1, col2 = st.columns(2)
    with col1:
        try:
            min_date = df[processor.date_column].min().date() if processor.date_column and not df.empty else (datetime.now() - timedelta(days=365)).date()
        except:
            min_date = (datetime.now() - timedelta(days=365)).date()
        
        start_date = st.date_input(
            "Fecha Inicio",
            value=min_date
        )
    with col2:
        try:
            max_date = df[processor.date_column].max().date() if processor.date_column and not df.empty else datetime.now().date()
        except:
            max_date = datetime.now().date()
            
        end_date = st.date_input(
            "Fecha Fin",
            value=max_date
        )
    
    # Filter data
    filtered_df = processor.filter_by_date_range(df, start_date, end_date)
    
    # KPI Cards
    create_kpi_cards(filtered_df, processor)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if processor.date_column and processor.amount_column:
            fig_trend = create_sales_trend_chart(filtered_df, processor)
            st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        if processor.salesperson_column and processor.amount_column:
            fig_team = create_team_performance_chart(filtered_df, processor)
            st.plotly_chart(fig_team, use_container_width=True)
    
    # Recent data table
    st.subheader("📋 Datos Recientes")
    st.dataframe(filtered_df.tail(10), use_container_width=True)

def show_team_analysis():
    """Display team analysis page"""
    from pages.team_analysis import render_team_analysis
    render_team_analysis(st.session_state.df, st.session_state.data_processor)

def show_individual_analysis():
    """Display individual analysis page"""
    from pages.individual_analysis import render_individual_analysis
    render_individual_analysis(st.session_state.df, st.session_state.data_processor)

def show_forecasting():
    """Display forecasting page"""
    from pages.forecasting_page import render_forecasting_page
    render_forecasting_page(st.session_state.df, st.session_state.data_processor)

if __name__ == "__main__":
    main()
