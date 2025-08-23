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
        self.contract_column = None
        
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
        # Enhanced date column patterns
        date_patterns = [
            'mes de gestión', 'mes de gestion', 'fecha_venta', 'fecha_cierre',
            'periodo', 'mes', 'año', 'fecha', 'date', 'time', 'timestamp'
        ]
        
        # Enhanced amount/sales patterns
        amount_patterns = [
            'venta uf', 'venta_uf', 'monto_uf', 'valor_uf', 'precio_uf',
            'monto', 'amount', 'venta', 'ventas', 'sale', 'sales',
            'precio', 'price', 'valor', 'value', 'revenue', 'ingreso'
        ]
        
        # Enhanced salesperson patterns
        salesperson_patterns = [
            'eevv', 'ejecutivo', 'asesor', 'vendedor', 'supervisor',
            'salesperson', 'representative', 'rep', 'agent', 'agente',
            'consultor', 'broker'
        ]
        
        # Product patterns
        product_patterns = [
            'producto', 'product', 'item', 'articulo', 'servicio', 'service',
            'tipo_producto', 'categoria_producto', 'linea', 'modelo'
        ]
        
        # Customer patterns
        customer_patterns = [
            'cliente', 'customer', 'client', 'rut_cliente', 'empresa',
            'company', 'razon_social', 'nombre_cliente'
        ]
        
        # Region patterns
        region_patterns = [
            'region', 'zona', 'area', 'territory', 'territorio',
            'ciudad', 'city', 'sucursal', 'oficina'
        ]
        
        # Contract patterns
        contract_patterns = [
            'contratos', 'contrato', 'contracts', 'contract',
            'num_contratos', 'cantidad_contratos', 'total_contratos'
        ]
        
        # Detect columns with priority scoring
        self._detect_with_priority(df, 'date_column', date_patterns, self._is_date_column)
        self._detect_with_priority(df, 'amount_column', amount_patterns, self._is_numeric_column)
        self._detect_with_priority(df, 'salesperson_column', salesperson_patterns, lambda x: True)
        self._detect_with_priority(df, 'product_column', product_patterns, lambda x: True)
        self._detect_with_priority(df, 'customer_column', customer_patterns, lambda x: True)
        self._detect_with_priority(df, 'region_column', region_patterns, lambda x: True)
        self._detect_with_priority(df, 'contract_column', contract_patterns, self._is_numeric_column)
    
    def _detect_with_priority(self, df, attr_name, patterns, validator):
        """Detect column with priority scoring"""
        best_match = None
        best_score = 0
        
        for col in df.columns:
            col_lower = col.lower().strip()
            for i, pattern in enumerate(patterns):
                if pattern in col_lower:
                    # Higher score for exact matches and earlier patterns
                    score = (len(patterns) - i) * 10
                    if col_lower == pattern:
                        score += 50  # Bonus for exact match
                    if col_lower.startswith(pattern) or col_lower.endswith(pattern):
                        score += 20  # Bonus for start/end match
                    
                    # Validate the column content
                    if validator(df[col]) and score > best_score:
                        best_match = col
                        best_score = score
                        break
        
        if best_match:
            setattr(self, attr_name, best_match)
        
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
        """Enhanced date column detection"""
        if series.empty:
            return False
            
        sample_values = series.dropna().head(20).astype(str)
        date_indicators = 0
        
        for val in sample_values:
            val_clean = val.strip().lower()
            
            # Try direct datetime conversion
            try:
                pd.to_datetime(val)
                date_indicators += 1
                continue
            except:
                pass
            
            # Check for month names
            month_names = [
                'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre',
                'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
            ]
            if any(month in val_clean for month in month_names):
                date_indicators += 1
                continue
            
            # Check for date patterns
            import re
            date_patterns = [
                r'\d{4}-\d{1,2}',  # YYYY-MM
                r'\d{1,2}/\d{4}',  # MM/YYYY
                r'\d{4}/\d{1,2}',  # YYYY/MM
                r'\d{1,2}-\d{4}',  # MM-YYYY
                r'\d{4}\d{2}',     # YYYYMM
            ]
            
            for pattern in date_patterns:
                if re.match(pattern, val_clean):
                    date_indicators += 1
                    break
        
        # Consider it a date column if >70% of values look like dates
        return date_indicators / len(sample_values) > 0.7
    
    def _is_numeric_column(self, series):
        """Enhanced numeric column detection"""
        if series.empty:
            return False
            
        sample_values = series.dropna().head(20)
        numeric_count = 0
        
        for val in sample_values:
            try:
                # Try direct numeric conversion
                pd.to_numeric(val)
                numeric_count += 1
            except:
                # Try cleaning common formats
                val_str = str(val).strip()
                # Remove common currency symbols and separators
                cleaned = val_str.replace('$', '').replace(',', '').replace('.', '').replace(' ', '')
                try:
                    float(cleaned)
                    numeric_count += 1
                except:
                    pass
        
        # Consider numeric if >80% of values are numeric
        return numeric_count / len(sample_values) > 0.8
    
    def _clean_data(self, df):
        """Enhanced data cleaning and processing"""
        df = df.copy()
        
        # Enhanced date column processing
        if self.date_column:
            df[self.date_column] = self._clean_date_column(df[self.date_column])
        
        # Enhanced amount column processing
        if self.amount_column:
            df[self.amount_column] = self._clean_amount_column(df[self.amount_column])
        
        # Clean text columns
        text_columns = [self.salesperson_column, self.product_column, self.customer_column, 
                       self.region_column, self.category_column]
        for col in text_columns:
            if col and col in df.columns:
                df[col] = self._clean_text_column(df[col])
        
        # Clean contract column if it exists
        if self.contract_column and self.contract_column in df.columns:
            df[self.contract_column] = self._clean_amount_column(df[self.contract_column])
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove rows with missing critical data
        critical_columns = [col for col in [self.date_column, self.amount_column] if col]
        if critical_columns:
            df = df.dropna(subset=critical_columns)
        
        # Sort by date if available
        if self.date_column:
            df = df.sort_values(self.date_column)
        
        return df
    
    def _clean_date_column(self, series):
        """Clean and convert date column with multiple format attempts"""
        # Try multiple date formats
        date_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%Y-%m', '%m/%Y', '%Y/%m', '%d-%m-%Y',
            '%B %Y', '%b %Y', '%Y %B', '%Y %b'
        ]
        
        result = pd.to_datetime(series, errors='coerce')
        
        # If many NaT values, try specific formats
        if result.isna().sum() > len(series) * 0.3:
            for fmt in date_formats:
                try:
                    temp_result = pd.to_datetime(series, format=fmt, errors='coerce')
                    if temp_result.isna().sum() < result.isna().sum():
                        result = temp_result
                except:
                    continue
        
        return result
    
    def _clean_amount_column(self, series):
        """Clean and convert amount column"""
        if series.dtype == 'object':
            # Remove currency symbols, spaces, and convert decimal separators
            cleaned = series.astype(str)
            cleaned = cleaned.str.replace(r'[^\d.,-]', '', regex=True)
            cleaned = cleaned.str.replace(',', '.')
            cleaned = cleaned.str.replace(r'\.(?=.*\.)', '', regex=True)  # Remove extra dots
        else:
            cleaned = series
        
        return pd.to_numeric(cleaned, errors='coerce')
    
    def _clean_text_column(self, series):
        """Clean text columns"""
        cleaned = series.astype(str)
        cleaned = cleaned.str.strip()
        cleaned = cleaned.str.title()  # Proper case
        cleaned = cleaned.replace('Nan', np.nan)
        return cleaned
    
    def get_detected_columns(self):
        """Return detected columns information with confidence scores"""
        return {
            'fecha': self.date_column,
            'monto': self.amount_column,
            'vendedor': self.salesperson_column,
            'producto': self.product_column,
            'cliente': self.customer_column,
            'region': self.region_column,
            'categoria': self.category_column,
            'contratos': self.contract_column
        }
    
    def get_data_quality_report(self, df):
        """Generate data quality report"""
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'data_types': {},
            'duplicates': df.duplicated().sum(),
            'date_range': None,
            'amount_stats': None
        }
        
        # Missing data analysis
        for col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            report['missing_data'][col] = f"{missing_pct:.1f}%"
        
        # Data types
        for col in df.columns:
            report['data_types'][col] = str(df[col].dtype)
        
        # Date range
        if self.date_column and self.date_column in df.columns:
            try:
                min_date = df[self.date_column].min()
                max_date = df[self.date_column].max()
                report['date_range'] = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            except:
                report['date_range'] = "Invalid date format"
        
        # Amount statistics
        if self.amount_column and self.amount_column in df.columns:
            try:
                stats = df[self.amount_column].describe()
                report['amount_stats'] = {
                    'min': f"{stats['min']:,.2f}",
                    'max': f"{stats['max']:,.2f}",
                    'mean': f"{stats['mean']:,.2f}",
                    'median': f"{stats['50%']:,.2f}"
                }
            except:
                report['amount_stats'] = "Invalid numeric format"
        
        return report
    
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
        
        monthly.columns = ['month', 'total_sales', 'contratos', 'avg_sale']
        monthly['month'] = monthly['month'].astype(str)
        
        return monthly
    
    def get_salesperson_performance(self, df):
        """Get performance metrics by salesperson"""
        if not self.salesperson_column or not self.amount_column:
            return pd.DataFrame()
        
        performance = df.groupby(self.salesperson_column)[self.amount_column].agg([
            'sum', 'count', 'mean', 'std'
        ]).reset_index()
        
        performance.columns = ['salesperson', 'total_sales', 'contratos', 'avg_sale', 'std_sale']
        performance = performance.sort_values('total_sales', ascending=False)
        
        return performance
