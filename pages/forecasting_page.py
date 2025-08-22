import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils.forecasting import SalesForecaster
from datetime import datetime, timedelta

def render_forecasting_page(df, data_processor):
    """Render sales forecasting page"""
    st.subheader("📈 Forecasting de Ventas")
    
    if df.empty:
        st.warning("No hay datos disponibles para el forecasting")
        return
    
    if not data_processor.date_column or not data_processor.amount_column:
        st.error("Se requieren columnas de fecha y monto para realizar el forecasting")
        return
    
    # Initialize forecaster
    forecaster = SalesForecaster(data_processor)
    
    # Forecasting parameters
    st.sidebar.subheader("⚙️ Parámetros de Forecasting")
    
    # Date range for historical data
    min_date = df[data_processor.date_column].min()
    max_date = df[data_processor.date_column].max()
    
    try:
        min_date_val = min_date.date() if hasattr(min_date, 'date') else min_date
        max_date_val = max_date.date() if hasattr(max_date, 'date') else max_date
    except:
        min_date_val = (datetime.now() - timedelta(days=365)).date()
        max_date_val = datetime.now().date()
    
    start_date = st.sidebar.date_input(
        "Fecha Inicio Histórica",
        value=min_date_val,
        min_value=min_date_val,
        max_value=max_date_val
    )
    
    end_date = st.sidebar.date_input(
        "Fecha Fin Histórica",
        value=max_date_val,
        min_value=min_date_val,
        max_value=max_date_val
    )
    
    # Forecast parameters
    forecast_periods = st.sidebar.slider(
        "Períodos a Pronosticar",
        min_value=1,
        max_value=12,
        value=6,
        help="Número de meses a pronosticar"
    )
    
    aggregation_level = st.sidebar.selectbox(
        "Nivel de Agregación",
        ["monthly", "weekly"],
        index=0,
        help="Nivel temporal para el análisis"
    )
    
    forecast_method = st.sidebar.selectbox(
        "Método de Forecasting",
        ["Linear Regression", "Moving Average", "Seasonal Decomposition"],
        index=0
    )
    
    # Salesperson filter for individual forecasting
    individual_forecast = st.sidebar.checkbox("Forecasting Individual por Vendedor")
    
    selected_salesperson = None
    if individual_forecast and data_processor.salesperson_column:
        salespeople = sorted(df[data_processor.salesperson_column].unique())
        selected_salesperson = st.sidebar.selectbox(
            "Seleccionar Vendedor",
            salespeople
        )
    
    # Filter data
    filtered_df = data_processor.filter_by_date_range(df, start_date, end_date)
    
    if individual_forecast and selected_salesperson:
        filtered_df = data_processor.filter_by_salesperson(filtered_df, selected_salesperson)
        forecast_title = f"Forecasting - {selected_salesperson}"
    else:
        forecast_title = "Forecasting - Equipo Completo"
    
    if filtered_df.empty:
        st.warning("No hay datos en el rango seleccionado")
        return
    
    try:
        # Prepare data for forecasting
        with st.spinner("Preparando datos para forecasting..."):
            df_agg = forecaster.prepare_forecasting_data(filtered_df, aggregation_level)
        
        if len(df_agg) < 3:
            st.error("Se necesitan al menos 3 períodos de datos históricos para realizar el forecasting")
            return
        
        # Display historical data overview
        st.subheader("📊 Datos Históricos")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📅 Períodos Históricos", len(df_agg))
        
        with col2:
            avg_sales = df_agg[data_processor.amount_column].mean()
            st.metric("💰 Venta Promedio", f"${avg_sales:,.2f}")
        
        with col3:
            total_sales = df_agg[data_processor.amount_column].sum()
            st.metric("📈 Ventas Totales", f"${total_sales:,.2f}")
        
        with col4:
            if len(df_agg) > 1:
                growth_rate = ((df_agg[data_processor.amount_column].iloc[-1] - df_agg[data_processor.amount_column].iloc[0]) / df_agg[data_processor.amount_column].iloc[0]) * 100
                st.metric("🚀 Crecimiento Total", f"{growth_rate:+.1f}%")
        
        # Generate forecast based on selected method
        with st.spinner(f"Generando forecasting con {forecast_method}..."):
            if forecast_method == "Linear Regression":
                forecast_df = forecaster.create_linear_regression_forecast(df_agg, forecast_periods)
            elif forecast_method == "Moving Average":
                window = min(3, len(df_agg))
                forecast_df = forecaster.create_moving_average_forecast(df_agg, forecast_periods, window)
            else:  # Seasonal Decomposition
                forecast_df = forecaster.create_seasonal_forecast(df_agg, forecast_periods)
        
        # Display forecast results
        st.subheader("🔮 Resultados del Forecasting")
        
        # Forecast metrics
        forecast_total = forecast_df['forecast'].sum()
        avg_forecast = forecast_df['forecast'].mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💰 Ventas Proyectadas", f"${forecast_total:,.2f}")
        
        with col2:
            st.metric("📊 Promedio Proyectado", f"${avg_forecast:,.2f}")
        
        with col3:
            if len(df_agg) > 0:
                historical_avg = df_agg[data_processor.amount_column].mean()
                forecast_vs_historical = ((avg_forecast - historical_avg) / historical_avg) * 100
                st.metric("📈 vs. Histórico", f"{forecast_vs_historical:+.1f}%")
        
        # Forecast chart
        fig_forecast = forecaster.create_forecast_chart(df_agg, forecast_df, forecast_title)
        st.plotly_chart(fig_forecast, use_container_width=True, key="forecast_main_chart")
        
        # Forecast table
        st.subheader("📋 Tabla de Pronósticos")
        
        display_forecast = forecast_df.copy()
        display_forecast[data_processor.date_column] = display_forecast[data_processor.date_column].dt.strftime('%Y-%m')
        display_forecast['forecast'] = display_forecast['forecast'].apply(lambda x: f"${x:,.2f}")
        display_forecast['lower_bound'] = display_forecast['lower_bound'].apply(lambda x: f"${x:,.2f}")
        display_forecast['upper_bound'] = display_forecast['upper_bound'].apply(lambda x: f"${x:,.2f}")
        
        display_forecast.columns = ['Período', 'Pronóstico', 'Límite Inferior', 'Límite Superior']
        st.dataframe(display_forecast, use_container_width=True)
        
        # Model evaluation
        if len(df_agg) >= 6:
            st.subheader("📊 Evaluación del Modelo")
            
            with st.spinner("Evaluando precisión del modelo..."):
                metrics = forecaster.evaluate_model_accuracy(df_agg)
            
            if "error" not in metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MAE", f"${metrics['mae']:,.2f}", help="Error Absoluto Medio")
                
                with col2:
                    st.metric("RMSE", f"${metrics['rmse']:,.2f}", help="Raíz del Error Cuadrático Medio")
                
                with col3:
                    st.metric("MAPE", f"{metrics['mape']:.1f}%", help="Error Porcentual Absoluto Medio")
                
                with col4:
                    st.metric("R²", f"{metrics['r2']:.3f}", help="Coeficiente de Determinación")
                
                # Model accuracy interpretation
                if metrics['mape'] < 10:
                    accuracy_level = "🟢 Excelente"
                elif metrics['mape'] < 20:
                    accuracy_level = "🟡 Buena"
                else:
                    accuracy_level = "🔴 Moderada"
                
                st.info(f"**Precisión del Modelo: {accuracy_level}** (MAPE: {metrics['mape']:.1f}%)")
        
        # Scenario analysis
        st.subheader("🎯 Análisis de Escenarios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Optimistic scenario (increase forecast by percentage)
            optimistic_factor = st.slider(
                "Escenario Optimista (%)",
                min_value=0,
                max_value=50,
                value=10,
                help="Incremento porcentual para escenario optimista"
            ) / 100
            
            optimistic_forecast = forecast_df['forecast'] * (1 + optimistic_factor)
            optimistic_total = optimistic_forecast.sum()
            
            st.metric(
                "🚀 Escenario Optimista",
                f"${optimistic_total:,.2f}",
                delta=f"+{optimistic_factor*100:.0f}%"
            )
        
        with col2:
            # Conservative scenario (decrease forecast by percentage)
            conservative_factor = st.slider(
                "Escenario Conservador (%)",
                min_value=0,
                max_value=30,
                value=10,
                help="Reducción porcentual para escenario conservador"
            ) / 100
            
            conservative_forecast = forecast_df['forecast'] * (1 - conservative_factor)
            conservative_total = conservative_forecast.sum()
            
            st.metric(
                "🛡️ Escenario Conservador",
                f"${conservative_total:,.2f}",
                delta=f"-{conservative_factor*100:.0f}%"
            )
        
        # Scenario comparison chart
        scenario_df = forecast_df.copy()
        scenario_df['optimistic'] = optimistic_forecast
        scenario_df['conservative'] = conservative_forecast
        
        fig_scenarios = go.Figure()
        
        # Add scenarios
        fig_scenarios.add_trace(go.Scatter(
            x=scenario_df[data_processor.date_column],
            y=scenario_df['conservative'],
            mode='lines',
            name='Conservador',
            line=dict(color='orange', dash='dot')
        ))
        
        fig_scenarios.add_trace(go.Scatter(
            x=scenario_df[data_processor.date_column],
            y=scenario_df['forecast'],
            mode='lines+markers',
            name='Pronóstico Base',
            line=dict(color='blue')
        ))
        
        fig_scenarios.add_trace(go.Scatter(
            x=scenario_df[data_processor.date_column],
            y=scenario_df['optimistic'],
            mode='lines',
            name='Optimista',
            line=dict(color='green', dash='dot')
        ))
        
        fig_scenarios.update_layout(
            title="📊 Comparación de Escenarios",
            xaxis_title="Fecha",
            yaxis_title="Ventas ($)",
            template='plotly_white'
        )
        
        st.plotly_chart(fig_scenarios, use_container_width=True, key="forecast_scenarios")
        
        # Monthly breakdown if forecasting team
        if not individual_forecast and data_processor.salesperson_column:
            st.subheader("👥 Forecasting por Vendedor")
            
            # Calculate individual contribution percentages
            salesperson_contrib = filtered_df.groupby(data_processor.salesperson_column)[data_processor.amount_column].sum()
            total_contrib = salesperson_contrib.sum()
            contrib_pct = salesperson_contrib / total_contrib
            
            # Apply to forecast
            individual_forecasts = {}
            for salesperson, pct in contrib_pct.items():
                individual_forecasts[salesperson] = forecast_df['forecast'] * pct
            
            # Create DataFrame for display
            individual_df = pd.DataFrame(individual_forecasts)
            individual_df['Total'] = individual_df.sum(axis=1)
            individual_df['Period'] = forecast_df[data_processor.date_column].dt.strftime('%Y-%m')
            
            # Display table
            display_individual = individual_df.set_index('Period')
            for col in display_individual.columns:
                display_individual[col] = display_individual[col].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(display_individual, use_container_width=True)
        
        # Export forecast
        st.subheader("📥 Exportar Forecasting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Descargar Pronósticos"):
                export_df = forecast_df.copy()
                export_df[data_processor.date_column] = export_df[data_processor.date_column].dt.strftime('%Y-%m-%d')
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📄 Descargar CSV",
                    data=csv,
                    file_name=f"forecast_{forecast_method.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Descargar Datos Históricos"):
                export_historical = df_agg[[data_processor.date_column, data_processor.amount_column]].copy()
                export_historical[data_processor.date_column] = export_historical[data_processor.date_column].dt.strftime('%Y-%m-%d')
                csv = export_historical.to_csv(index=False)
                
                st.download_button(
                    label="📄 Descargar Históricos CSV",
                    data=csv,
                    file_name="datos_historicos.csv",
                    mime="text/csv"
                )
        
        # Recommendations
        st.subheader("💡 Recomendaciones")
        
        # Calculate trends
        if len(df_agg) >= 3:
            recent_trend = np.polyfit(range(3), df_agg[data_processor.amount_column].tail(3), 1)[0]
            forecast_trend = np.polyfit(range(len(forecast_df)), forecast_df['forecast'], 1)[0]
            
            recommendations = []
            
            if forecast_trend > recent_trend * 1.1:
                recommendations.append("🟢 El modelo proyecta una mejora en las ventas. Considera aumentar el inventario y preparar estrategias de crecimiento.")
            elif forecast_trend < recent_trend * 0.9:
                recommendations.append("🟡 Se proyecta una desaceleración. Evalúa estrategias de marketing y promociones para impulsar las ventas.")
            else:
                recommendations.append("🔵 Las ventas se mantendrán estables. Mantén las estrategias actuales y optimiza la eficiencia operativa.")
            
            if individual_forecast and selected_salesperson:
                team_performance = df.groupby(data_processor.salesperson_column)[data_processor.amount_column].sum()
                salesperson_rank = team_performance.rank(ascending=False)[selected_salesperson]
                
                if salesperson_rank <= len(team_performance) * 0.3:
                    recommendations.append(f"🏆 {selected_salesperson} está entre los mejores vendedores. Considera asignar clientes clave y objetivos ambiciosos.")
                else:
                    recommendations.append(f"📈 {selected_salesperson} tiene potencial de mejora. Considera capacitación adicional y mentoring.")
            
            for rec in recommendations:
                st.info(rec)
    
    except Exception as e:
        st.error(f"Error al generar el forecasting: {str(e)}")
        st.info("Verifica que los datos tengan el formato correcto y suficiente información histórica.")
