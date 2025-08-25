import streamlit as st
import pandas as pd
from utils.ml_predictions import (SalesPredictionEngine, create_sales_prediction_chart, 
                                 create_top_performer_probability_chart, create_churn_risk_chart)

def render_ml_predictions_page(df, data_processor):
    """Render ML predictions page"""
    st.subheader("🤖 Predicciones Avanzadas con IA")
    
    if df.empty:
        st.warning("No hay datos disponibles para predicciones")
        return
    
    # Initialize prediction engine
    if 'prediction_engine' not in st.session_state:
        st.session_state.prediction_engine = SalesPredictionEngine()
    
    engine = st.session_state.prediction_engine
    
    # Tabs for different predictions
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Predicción de Ventas",
        "🏆 Top Performers",
        "⚠️ Riesgo de Clientes",
        "📊 Insights IA"
    ])
    
    with tab1:
        st.subheader("🔮 Predicción de Ventas - Próximo Mes")
        
        with st.spinner("Entrenando modelo de predicción..."):
            predictions = engine.predict_next_month_sales(df, data_processor)
        
        if predictions:
            # Show predictions chart
            fig = create_sales_prediction_chart(predictions)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show predictions table
            st.subheader("📋 Predicciones Detalladas")
            
            pred_df = pd.DataFrame(predictions)
            pred_df['prediccion_formatted'] = pred_df['prediccion'].apply(lambda x: f"{x:,.0f} UF".replace(',', '.'))
            pred_df['ultimo_mes_formatted'] = pred_df['ultimo_mes'].apply(lambda x: f"{x:,.0f} UF".replace(',', '.'))
            pred_df['cambio'] = ((pred_df['prediccion'] - pred_df['ultimo_mes']) / pred_df['ultimo_mes'] * 100).round(1)
            
            display_df = pred_df[['vendedor', 'prediccion_formatted', 'ultimo_mes_formatted', 'cambio', 'confianza']].copy()
            
            st.dataframe(
                display_df,
                column_config={
                    "vendedor": st.column_config.TextColumn("👤 Vendedor"),
                    "prediccion_formatted": st.column_config.TextColumn("🔮 Predicción"),
                    "ultimo_mes_formatted": st.column_config.TextColumn("📊 Último Mes"),
                    "cambio": st.column_config.NumberColumn("📈 Cambio (%)", format="%.1f%%"),
                    "confianza": st.column_config.NumberColumn("🎯 Confianza (%)", format="%.0f%%")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Summary insights
            total_predicted = sum([p['prediccion'] for p in predictions])
            total_last_month = sum([p['ultimo_mes'] for p in predictions])
            growth_rate = ((total_predicted - total_last_month) / total_last_month * 100) if total_last_month > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Total Predicho", f"{total_predicted:,.0f} UF".replace(',', '.'))
            with col2:
                st.metric("📊 Último Mes Real", f"{total_last_month:,.0f} UF".replace(',', '.'))
            with col3:
                st.metric("📈 Crecimiento Esperado", f"{growth_rate:+.1f}%")
        
        else:
            st.warning("No hay suficientes datos históricos para generar predicciones confiables (mínimo 3 meses)")
    
    with tab2:
        st.subheader("🏆 Probabilidad de Top Performers")
        
        with st.spinner("Analizando patrones de rendimiento..."):
            top_performer_predictions = engine.predict_top_performer_probability(df, data_processor)
        
        if top_performer_predictions:
            # Show probability chart
            fig = create_top_performer_probability_chart(top_performer_predictions)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed probabilities
            st.subheader("📋 Probabilidades Detalladas")
            
            prob_df = pd.DataFrame(top_performer_predictions)
            prob_df['ventas_recientes_formatted'] = prob_df['ventas_recientes'].apply(lambda x: f"{x:,.0f} UF".replace(',', '.'))
            
            # Add risk categories
            def get_category(prob):
                if prob >= 70:
                    return "🌟 Muy Alto"
                elif prob >= 50:
                    return "⭐ Alto"
                elif prob >= 30:
                    return "📈 Medio"
                else:
                    return "📉 Bajo"
            
            prob_df['categoria'] = prob_df['probabilidad'].apply(get_category)
            
            display_df = prob_df[['vendedor', 'probabilidad', 'categoria', 'ventas_recientes_formatted']].copy()
            
            st.dataframe(
                display_df,
                column_config={
                    "vendedor": st.column_config.TextColumn("👤 Vendedor"),
                    "probabilidad": st.column_config.NumberColumn("🎯 Probabilidad (%)", format="%.1f%%"),
                    "categoria": st.column_config.TextColumn("📊 Categoría"),
                    "ventas_recientes_formatted": st.column_config.TextColumn("💰 Ventas Recientes")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Insights
            high_prob = [p for p in top_performer_predictions if p['probabilidad'] >= 70]
            if high_prob:
                st.success(f"🌟 {len(high_prob)} vendedores tienen alta probabilidad (≥70%) de ser top performers")
            
            low_prob = [p for p in top_performer_predictions if p['probabilidad'] < 30]
            if low_prob:
                st.warning(f"📉 {len(low_prob)} vendedores necesitan atención especial (probabilidad <30%)")
        
        else:
            st.warning("No hay suficientes datos para calcular probabilidades de top performers")
    
    with tab3:
        st.subheader("⚠️ Análisis de Riesgo de Clientes")
        
        if not data_processor.customer_column:
            st.warning("No se detectó columna de clientes. Esta funcionalidad requiere datos de clientes.")
            return
        
        with st.spinner("Analizando riesgo de pérdida de clientes..."):
            churn_predictions = engine.predict_customer_churn_risk(df, data_processor)
        
        if churn_predictions:
            # Show churn risk chart
            fig = create_churn_risk_chart(churn_predictions)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show high-risk customers table
            st.subheader("🚨 Clientes de Alto Riesgo")
            
            churn_df = pd.DataFrame(churn_predictions)
            churn_df['total_sales_formatted'] = churn_df['total_sales'].apply(lambda x: f"{x:,.0f} UF".replace(',', '.'))
            
            # Add risk level
            def get_risk_level(risk):
                if risk >= 80:
                    return "🔴 Crítico"
                elif risk >= 60:
                    return "🟠 Alto"
                elif risk >= 40:
                    return "🟡 Medio"
                else:
                    return "🟢 Bajo"
            
            churn_df['nivel_riesgo'] = churn_df['churn_risk'].apply(get_risk_level)
            
            display_df = churn_df[['customer', 'churn_risk', 'nivel_riesgo', 'recency', 'total_sales_formatted', 'frequency']].copy()
            
            st.dataframe(
                display_df,
                column_config={
                    "customer": st.column_config.TextColumn("🏢 Cliente"),
                    "churn_risk": st.column_config.NumberColumn("⚠️ Riesgo (%)", format="%.0f%%"),
                    "nivel_riesgo": st.column_config.TextColumn("📊 Nivel"),
                    "recency": st.column_config.NumberColumn("📅 Días sin Comprar", format="%.0f"),
                    "total_sales_formatted": st.column_config.TextColumn("💰 Ventas Totales"),
                    "frequency": st.column_config.NumberColumn("🔄 Frecuencia", format="%.0f")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Action recommendations
            critical_customers = [c for c in churn_predictions if c['churn_risk'] >= 80]
            if critical_customers:
                st.error(f"🚨 {len(critical_customers)} clientes en riesgo crítico requieren atención inmediata")
                
                st.subheader("💡 Recomendaciones de Acción")
                for customer in critical_customers[:3]:  # Show top 3
                    with st.expander(f"📋 Plan para {customer['customer']}"):
                        st.write(f"**Riesgo:** {customer['churn_risk']:.0f}%")
                        st.write(f"**Días sin comprar:** {customer['recency']} días")
                        st.write(f"**Valor del cliente:** {customer['total_sales']:,.0f} UF".replace(',', '.'))
                        
                        st.write("**Acciones recomendadas:**")
                        if customer['recency'] > 90:
                            st.write("• 📞 Contacto inmediato para reactivación")
                        if customer['frequency'] < 3:
                            st.write("• 🎁 Oferta especial personalizada")
                        if customer['total_sales'] > 10000:
                            st.write("• 👨‍💼 Asignar ejecutivo senior")
                        st.write("• 📊 Análisis detallado de necesidades")
        
        else:
            st.warning("No hay suficientes datos de clientes para análisis de churn")
    
    with tab4:
        st.subheader("🧠 Insights de Inteligencia Artificial")
        
        # General AI insights
        if not df.empty and data_processor.salesperson_column and data_processor.amount_column:
            
            # Performance patterns
            st.subheader("🔍 Patrones Identificados")
            
            # Seasonality analysis
            if data_processor.date_column:
                monthly_sales = df.groupby(df[data_processor.date_column].dt.month)[data_processor.amount_column].mean()
                best_month = monthly_sales.idxmax()
                worst_month = monthly_sales.idxmin()
                
                month_names = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
                              7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
                
                st.info(f"📅 **Estacionalidad**: {month_names[best_month]} es el mejor mes, {month_names[worst_month]} el más bajo")
            
            # Performance consistency
            salesperson_cv = df.groupby(data_processor.salesperson_column)[data_processor.amount_column].agg(['mean', 'std'])
            salesperson_cv['cv'] = salesperson_cv['std'] / salesperson_cv['mean']
            most_consistent = salesperson_cv['cv'].idxmin()
            most_volatile = salesperson_cv['cv'].idxmax()
            
            st.info(f"📊 **Consistencia**: {most_consistent} es el más consistente, {most_volatile} el más volátil")
            
            # Growth trends
            if data_processor.date_column and len(df) > 30:
                recent_data = df.tail(len(df)//2)
                older_data = df.head(len(df)//2)
                
                recent_avg = recent_data[data_processor.amount_column].mean()
                older_avg = older_data[data_processor.amount_column].mean()
                
                if recent_avg > older_avg * 1.1:
                    st.success("📈 **Tendencia**: Las ventas muestran crecimiento sostenido")
                elif recent_avg < older_avg * 0.9:
                    st.warning("📉 **Tendencia**: Las ventas muestran declive, requiere atención")
                else:
                    st.info("📊 **Tendencia**: Las ventas se mantienen estables")
            
            # Recommendations
            st.subheader("💡 Recomendaciones Estratégicas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🎯 Optimización de Ventas:**
                • Enfocar esfuerzos en meses de alta estacionalidad
                • Implementar programas de coaching para vendedores volátiles
                • Replicar estrategias de vendedores consistentes
                """)
            
            with col2:
                st.markdown("""
                **🔄 Retención de Clientes:**
                • Programa de seguimiento automático cada 30 días
                • Ofertas personalizadas para clientes de alto valor
                • Sistema de alertas tempranas de riesgo
                """)
        
        else:
            st.warning("Datos insuficientes para generar insights de IA")