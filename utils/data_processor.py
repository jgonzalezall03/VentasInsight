import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class DataProcessor:
    def __init__(self):
        self.date_column = None
        self.amount_column = None
        self.salesperson_column = None
        self.product_column = None
        self.customer_column = None
        self.region_column = None
        self.category_column = None
        
    def load_and_process_excel(self, uploaded_file):
        """Load and process Excel file"""
        try:
            # Try to read the Excel file
            df = pd.read_excel(uploaded_file)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Detect column types
            self._detect_column_types(df)
            
            # Clean and process data
            df = self._clean_data(df)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error al procesar el archivo Excel: {str(e)}")
    
    def _detect_column_types(self, df):
        """Automatically detect column types based on common patterns"""
        columns = df.columns.str.lower()
        
        # Date column patterns - Adding specific "Mes de gestión" pattern
        date_patterns = ['mes de gestión', 'mes de gestion', 'mes', 'fecha', 'date', 'tiempo', 'time', 'periodo', 'period']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in date_patterns):
                if self._is_date_column(df[col]) or 'mes' in col.lower():
                    self.date_column = col
                    # Prioritize "Mes de gestión" if found
                    if 'mes de gestión' in col.lower() or 'mes de gestion' in col.lower():
                        break
        
        # Amount/Sales column patterns - Adding specific "Venta UF" pattern
        amount_patterns = ['venta uf', 'monto', 'amount', 'venta', 'sale', 'precio', 'price', 'valor', 'value', 'revenue', 'ingreso', 'contratos']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in amount_patterns):
                if self._is_numeric_column(df[col]):
                    self.amount_column = col
                    # Prioritize "Venta UF" if found
                    if 'venta uf' in col.lower():
                        break
        
        # Salesperson column patterns - Adding specific "EEVV" and "Supervisor" patterns
        salesperson_patterns = ['eevv', 'supervisor / jz', 'supervisor', 'vendedor', 'salesperson', 'ejecutivo', 'representative', 'rep', 'agent', 'agente']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in salesperson_patterns):
                self.salesperson_column = col
                # Prioritize "EEVV" for individual analysis
                if col.lower() == 'eevv':
                    break
        
        # Product column patterns
        product_patterns = ['producto', 'product', 'item', 'articulo', 'servicio', 'service']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in product_patterns):
                self.product_column = col
                break
        
        # Customer column patterns
        customer_patterns = ['cliente', 'customer', 'client', 'company', 'empresa']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in customer_patterns):
                self.customer_column = col
                break
        
        # Region column patterns
        region_patterns = ['region', 'zona', 'area', 'territory', 'territorio', 'ciudad', 'city']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in region_patterns):
                self.region_column = col
                break
        
        # Category column patterns
        category_patterns = ['categoria', 'category', 'tipo', 'type', 'clase', 'class']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in category_patterns):
                self.category_column = col
                break
    
    def _is_date_column(self, series):
        """Check if a column contains date data"""
        try:
            # First try to convert to datetime
            pd.to_datetime(series.dropna().head(10))
            return True
        except:
            # If it fails, check if it's a month column with text like "2024-01", "Enero 2024", etc.
            sample_values = series.dropna().head(10).astype(str)
            # Check for patterns like "YYYY-MM", "Mes YYYY", "MM/YYYY", etc.
            for val in sample_values:
                if any(pattern in val.lower() for pattern in ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                                                             'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre',
                                                             'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                             'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                    return True
                # Check for YYYY-MM pattern
                if len(val.split('-')) == 2 and val.split('-')[0].isdigit() and val.split('-')[1].isdigit():
                    return True
                # Check for MM/YYYY pattern
                if len(val.split('/')) == 2 and all(part.isdigit() for part in val.split('/')):
                    return True
            return False
    
    def _is_numeric_column(self, series):
        """Check if a column contains numeric data"""
        try:
            pd.to_numeric(series.dropna().head(10))
            return True
        except:
            return False
    
    def _clean_data(self, df):
        """Clean and process the data"""
        # Convert date column
        if self.date_column:
            try:
                df[self.date_column] = pd.to_datetime(df[self.date_column], errors='coerce')
            except:
                # If conversion fails, try to handle month text data
                try:
                    # Handle cases like "Enero 2024", "2024-01", etc.
                    df[self.date_column] = pd.to_datetime(df[self.date_column], format='%Y-%m', errors='coerce')
                    if df[self.date_column].isna().all():
                        # Try other formats
                        df[self.date_column] = pd.to_datetime(df[self.date_column], format='%m/%Y', errors='coerce')
                except:
                    pass
        
        # Convert amount column
        if self.amount_column:
            # Remove currency symbols and convert to numeric
            if df[self.amount_column].dtype == 'object':
                df[self.amount_column] = df[self.amount_column].astype(str).str.replace(r'[^\d.-]', '', regex=True)
            df[self.amount_column] = pd.to_numeric(df[self.amount_column], errors='coerce')
        
        # Clean text columns
        text_columns = [self.salesperson_column, self.product_column, self.customer_column, 
                       self.region_column, self.category_column]
        for col in text_columns:
            if col and col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Remove rows with missing critical data
        critical_columns = [col for col in [self.date_column, self.amount_column] if col]
        if critical_columns:
            df = df.dropna(subset=critical_columns)
        
        # Sort by date if available
        if self.date_column:
            df = df.sort_values(self.date_column)
        
        return df
    
    def get_detected_columns(self):
        """Return detected columns information"""
        return {
            'fecha': self.date_column,
            'monto': self.amount_column,
            'vendedor': self.salesperson_column,
            'producto': self.product_column,
            'cliente': self.customer_column,
            'region': self.region_column,
            'categoria': self.category_column
        }
    
    def filter_by_date_range(self, df, start_date, end_date):
        """Filter dataframe by date range"""
        if not self.date_column:
            return df
        
        mask = (df[self.date_column].dt.date >= start_date) & (df[self.date_column].dt.date <= end_date)
        return df[mask]
    
    def filter_by_salesperson(self, df, salesperson):
        """Filter dataframe by salesperson"""
        if not self.salesperson_column or not salesperson:
            return df
        return df[df[self.salesperson_column] == salesperson]
    
    def get_sales_summary(self, df):
        """Get sales summary statistics"""
        if not self.amount_column:
            return {}
        
        total_sales = df[self.amount_column].sum()
        avg_sale = df[self.amount_column].mean()
        num_transactions = len(df)
        
        summary = {
            'total_sales': total_sales,
            'average_sale': avg_sale,
            'num_transactions': num_transactions,
            'max_sale': df[self.amount_column].max(),
            'min_sale': df[self.amount_column].min()
        }
        
        return summary
    
    def get_monthly_sales(self, df):
        """Get monthly sales aggregation"""
        if not self.date_column or not self.amount_column:
            return pd.DataFrame()
        
        monthly = df.groupby(df[self.date_column].dt.to_period('M'))[self.amount_column].agg([
            'sum', 'count', 'mean'
        ]).reset_index()
        
        monthly.columns = ['month', 'total_sales', 'transactions', 'avg_sale']
        monthly['month'] = monthly['month'].astype(str)
        
        return monthly
    
    def get_salesperson_performance(self, df):
        """Get performance metrics by salesperson"""
        if not self.salesperson_column or not self.amount_column:
            return pd.DataFrame()
        
        performance = df.groupby(self.salesperson_column)[self.amount_column].agg([
            'sum', 'count', 'mean', 'std'
        ]).reset_index()
        
        performance.columns = ['salesperson', 'total_sales', 'transactions', 'avg_sale', 'std_sale']
        performance = performance.sort_values('total_sales', ascending=False)
        
        return performance
