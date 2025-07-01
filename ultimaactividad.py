import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import time

# Machine Learning imports
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("Para usar Machine Learning, instala: pip install scikit-learn")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# --- FUNCIONES DE OBTENCIÓN DE DATOS Y FILTROS ---

def get_provincias_default():
    """Lista predefinida de respaldo para las provincias de Ecuador."""
    return [provincia.upper() for provincia in ['Azuay', 'Bolivar', 'Cañar', 'Carchi', 'Chimborazo', 'Cotopaxi',
            'El Oro', 'Esmeraldas', 'Galapagos', 'Guayas', 'Imbabura', 'Loja',
            'Los Rios', 'Manabi', 'Morona Santiago', 'Napo', 'Orellana', 'Pastaza',
            'Pichincha', 'Santa Elena', 'Santo Domingo De Los Tsachilas', 'Sucumbios', 'Tungurahua',
            'Zamora Chinchipe']]

@st.cache_data(show_spinner=False)
def get_provincias_from_api():
    """Obtiene la lista de provincias/entidades directamente desde la API."""
    try:
        url = "https://datosabiertos.compraspublicas.gob.ec/PLATAFORMA/api/get_analysis?year=2024"
        response = requests.get(url, timeout=25)
        response.raise_for_status()
        data = response.json()
    except Exception:
        pass
    return get_provincias_default()

def get_tipos_contrato_default():
    """Lista predefinida de respaldo para los tipos de contrato."""
    return ['Subasta Inversa Electrónica', 'Repuestos o Accesorios',
            'Obra artística, científica o literaria',
            'Menor Cuantía','Lista corta','Licitación de Seguros', 'Cotización',
            'Contratos entre Entidades Públicas o sus subsidiarias',
            'Contratacion directa',
            'Comunicación Social - Contratación Directa',
            'Catálogo electrónico - Mejor oferta',
            'Catálogo electrónico - Gran compra mejor oferta',
            'Catálogo electrónico - Compra directa',
            'Bienes y Servicios únicos',
            'Transporte de correo interno o internacional',
            'Licitación',
            'Contratación Directa por Terminación Unilateral',
            'Concurso publico',
            'Catálogo electrónico - Gran compra puja',
            'Asesoría y Patrocinio Jurídico',
            'Contrataciones con empresas públicas internacionales',
            'Comunicación Social - Proceso de Selección']

@st.cache_data(show_spinner=False)
def get_tipos_contrato_from_api():
    """Obtiene la lista de tipos de contrato directamente desde la API."""
    try:
        url = "https://datosabiertos.compraspublicas.gob.ec/PLATAFORMA/api/get_analysis?year=2024"
        response = requests.get(url, timeout=25)
        response.raise_for_status()
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            if 'internal_type' in df.columns:
                tipos = df['internal_type'].dropna().unique()
                return sorted([str(t) for t in tipos if str(t).strip()])
    except Exception:
        pass
    return get_tipos_contrato_default()

def clean_and_prepare_df(df):
    """Función auxiliar para procesar un DataFrame."""
    for col in ['contracts', 'total']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    if 'month' in df.columns:
        df['month'] = pd.to_numeric(df['month'], errors='coerce')
        meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        df['mes_nombre'] = df['month'].apply(lambda x: meses[int(x-1)] if pd.notna(x) and 1 <= x <= 12 else 'N/A')

    if 'province' not in df.columns and 'region' in df.columns:
        df.rename(columns={'region': 'province'}, inplace=True)
    elif 'province' not in df.columns and 'region' not in df.columns:
        df['province'] = 'Unknown'
        
    return df

@st.cache_data
def load_data(year, provincia, tipo_contrato):
    """Carga datos desde la API según los filtros."""
    url = "https://datosabiertos.compraspublicas.gob.ec/PLATAFORMA/api/get_analysis"
    params = {}
    if year and year != 'Todos': params["year"] = year
    if provincia and provincia != 'Todas': params["region"] = provincia.upper()
    if tipo_contrato and tipo_contrato != 'Todas': params["type"] = tipo_contrato
    try:
        response = requests.get(url, params=params, timeout=45)
        response.raise_for_status()
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            df = clean_and_prepare_df(df)
            return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión con la API: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error procesando los datos: {e}")
        return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data
def load_all_data(years):
    """Carga datos de todos los años, provincias y tipos."""
    all_dataframes = []
    url = "https://datosabiertos.compraspublicas.gob.ec/PLATAFORMA/api/get_analysis"
    progress_bar_placeholder = st.sidebar.empty()
    status_text_placeholder = st.sidebar.empty()
    progress_bar = progress_bar_placeholder.progress(0)
    
    for i, year in enumerate(years):
        status_text_placeholder.text(f"Consultando año {year}...")
        try:
            params = {"year": year}
            response = requests.get(url, params=params, timeout=90)
            response.raise_for_status()
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                df['year'] = year
                all_dataframes.append(df)
        except requests.exceptions.RequestException as e:
            st.warning(f"No se pudieron cargar los datos para el año {year}. Error: {e}")
        except Exception as e:
            st.warning(f"Ocurrió un error inesperado para el año {year}. Error: {e}")
        progress_bar.progress((i + 1) / len(years))
        time.sleep(1) 

    progress_bar_placeholder.empty()
    status_text_placeholder.empty()
    if not all_dataframes: return pd.DataFrame()
    full_df = pd.concat(all_dataframes, ignore_index=True)
    full_df = clean_and_prepare_df(full_df)
    return full_df

# --- FUNCIONES PARA CREAR GRÁFICAS ---
def create_contracts_by_month_and_type(df):
    if df.empty or not {'mes_nombre', 'internal_type', 'contracts'}.issubset(df.columns): return None
    df_grouped = df.groupby(['month', 'mes_nombre', 'internal_type'])['contracts'].sum().reset_index().sort_values('month')
    fig = px.bar(df_grouped, x='mes_nombre', y='contracts', color='internal_type', title='Total de Contratos por Mes y Tipo', labels={'mes_nombre': 'Mes', 'contracts': 'Número de Contratos', 'internal_type': 'Tipo de Contrato'}, color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_layout(xaxis_tickangle=-45, showlegend=True, height=500)
    return fig

def create_contracts_by_type(df):
    if df.empty or 'internal_type' not in df.columns or 'contracts' not in df.columns: return None
    df_grouped = df.groupby('internal_type')['contracts'].sum().reset_index()
    fig = px.bar(df_grouped, x='contracts', y='internal_type', orientation='h', title='Total de Contratos por Tipo', labels={'contracts': 'Número de Contratos', 'internal_type': 'Tipo de Contrato'}, color='contracts', color_continuous_scale='Greens')
    fig.update_layout(height=max(400, len(df_grouped) * 25), yaxis={'categoryorder':'total ascending'})
    return fig

def create_bar_chart_totals_by_type(df):
    if df.empty or 'internal_type' not in df.columns or 'contracts' not in df.columns: return None
    df_grouped = df.groupby('internal_type')['contracts'].sum().reset_index().sort_values('contracts', ascending=False)
    fig = px.bar(df_grouped, x='internal_type', y='contracts', title='Totales por Tipo de Contratación', labels={'internal_type': 'Tipo de Contrato', 'contracts': 'Total de Contratos'}, color='contracts', color_continuous_scale='Blues')
    fig.update_layout(xaxis_tickangle=-45, height=500)
    return fig

def create_line_chart_monthly_amounts(df):
    if df.empty or not {'mes_nombre', 'month', 'total'}.issubset(df.columns): return None
    df_monthly = df.groupby(['month', 'mes_nombre'])['total'].sum().reset_index().sort_values('month')
    fig = px.line(df_monthly, x='mes_nombre', y='total', title='Evolución Mensual de Montos Totales', labels={'mes_nombre': 'Mes', 'total': 'Monto Total (USD)'}, markers=True)
    fig.update_layout(xaxis_tickangle=-45, height=400)
    return fig

def create_pie_chart_contracts_by_type(df):
    if df.empty or 'internal_type' not in df.columns or 'contracts' not in df.columns: return None
    df_grouped = df.groupby('internal_type')['contracts'].sum().reset_index()
    fig = px.pie(df_grouped, values='contracts', names='internal_type', title='Proporción de Contratos por Tipo')
    fig.update_layout(height=500)
    return fig

def create_scatter_plot_total_vs_contracts(df):
    if df.empty or not {'contracts', 'total', 'internal_type'}.issubset(df.columns): return None
    df_scatter = df.groupby('internal_type').agg({'contracts': 'sum', 'total': 'sum'}).reset_index()
    fig = px.scatter(df_scatter, x='contracts', y='total', color='internal_type', title='Relación entre Total y Cantidad de Contratos', labels={'contracts': 'Número de Contratos', 'total': 'Monto Total (USD)', 'internal_type': 'Tipo de Contrato'})
    fig.update_layout(height=500)
    return fig

def create_line_chart_contracts_by_month_type(df):
    if df.empty or not {'mes_nombre', 'month', 'internal_type', 'contracts'}.issubset(df.columns): return None
    df_monthly_type = df.groupby(['month', 'mes_nombre', 'internal_type'])['contracts'].sum().reset_index().sort_values('month')
    fig = px.line(df_monthly_type, x='mes_nombre', y='contracts', color='internal_type', title='Tipos de Contrato por Mes', labels={'mes_nombre': 'Mes', 'contracts': 'Número de Contratos', 'internal_type': 'Tipo de Contrato'}, markers=True)
    fig.update_layout(xaxis_tickangle=-45, height=500)
    return fig

# --- FUNCIONES DE MACHINE LEARNING ---
def prepare_ml_data(df):
    if df.empty or not ML_AVAILABLE: return None
    try:
        if 'province' not in df.columns: df['province'] = 'Unknown'
        ml_df = df.groupby(['province', 'internal_type']).agg({'contracts': 'sum', 'total': 'sum', 'month': 'nunique'}).reset_index()
        ml_df['avg_contract_value'] = ml_df['total'] / (ml_df['contracts'] + 1)
        ml_df['contracts_per_month'] = ml_df['contracts'] / (ml_df['month'] + 1)
        return ml_df
    except Exception as e:
        st.error(f"Error preparando datos ML: {e}")
        return None

def apply_kmeans_clustering(df):
    if not ML_AVAILABLE: return
    ml_df = prepare_ml_data(df)
    if ml_df is None or len(ml_df) < 3: return
    try:
        features = ['contracts', 'total', 'avg_contract_value', 'contracts_per_month']
        X = ml_df[features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_clusters = min(4, len(ml_df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        ml_df['cluster'] = clusters
        fig = px.scatter(ml_df, x='contracts', y='total', color='cluster', hover_data=['province', 'internal_type'], title=f'Clustering K-Means - {n_clusters} Grupos', labels={'contracts': 'Contratos', 'total': 'Total USD'}, color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Resultado del Clustering:**")
        st.dataframe(ml_df.groupby('cluster').agg({'contracts': 'mean', 'total': 'mean', 'province': 'count'}).round(2))
    except Exception as e:
        st.error(f"Error en clustering: {e}")

def apply_prophet_forecasting(df):
    if not PROPHET_AVAILABLE: return
    if df.empty or 'month' not in df.columns: return
    try:
        monthly_data = df.groupby('month')['contracts'].sum().reset_index()
        if len(monthly_data) < 3:
            st.warning("Se necesitan al menos 3 meses de datos para la predicción con Prophet.")
            return
        monthly_data['ds'] = pd.to_datetime('2024-' + monthly_data['month'].astype(str) + '-01')
        monthly_data['y'] = monthly_data['contracts']
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        model.fit(monthly_data[['ds', 'y']])
        future = model.make_future_dataframe(periods=6, freq='M')
        forecast = model.predict(future)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_data['ds'], y=monthly_data['y'], mode='lines+markers', name='Histórico', line=dict(color='blue')))
        future_idx = len(monthly_data)
        fig.add_trace(go.Scatter(x=forecast['ds'][future_idx:], y=forecast['yhat'][future_idx:], mode='lines+markers', name='Predicción', line=dict(color='red', dash='dash')))
        fig.update_layout(title='Predicción de Contratos (Prophet)', xaxis_title='Fecha', yaxis_title='Contratos', height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Predicciones futuras:**")
        st.dataframe(forecast[['ds', 'yhat']][future_idx:].round(0))
    except Exception as e:
        st.error(f"Error en predicción con Prophet: {e}")

def apply_logistic_regression(df):
    if not ML_AVAILABLE: return
    ml_df = prepare_ml_data(df)
    if ml_df is None or len(ml_df) < 10: return
    try:
        threshold = ml_df['total'].median()
        ml_df['high_value'] = (ml_df['total'] > threshold).astype(int)
        features = ['contracts', 'avg_contract_value', 'contracts_per_month']
        X = ml_df[features].fillna(0)
        y = ml_df['high_value']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        importance = np.abs(model.coef_[0])
        fig = px.bar(x=features, y=importance, title=f'Importancia de Features - Precisión: {accuracy:.2%}', labels={'x': 'Features', 'y': 'Importancia'})
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"**Precisión del modelo de clasificación:** {accuracy:.2%}")
    except Exception as e:
        st.error(f"Error en Regresión Logística: {e}")

def apply_pca_analysis(df):
    if not ML_AVAILABLE: return
    ml_df = prepare_ml_data(df)
    if ml_df is None or len(ml_df) < 5: return
    try:
        features = ['contracts', 'total', 'avg_contract_value', 'contracts_per_month']
        X = ml_df[features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        ml_df['PC1'] = X_pca[:, 0]
        ml_df['PC2'] = X_pca[:, 1]
        fig = px.scatter(ml_df, x='PC1', y='PC2', color='internal_type', hover_data=['province'], title=f'Análisis de Componentes Principales (PCA) - Varianza: {pca.explained_variance_ratio_.sum():.1%}', labels={'PC1': f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%})', 'PC2': f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%})'})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error en PCA: {e}")

def apply_anomaly_detection(df):
    if not ML_AVAILABLE: return
    ml_df = prepare_ml_data(df)
    if ml_df is None or len(ml_df) < 10: return
    try:
        features = ['contracts', 'total', 'avg_contract_value', 'contracts_per_month']
        X = ml_df[features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X_scaled)
        ml_df['is_anomaly'] = ['Anomalía' if x == -1 else 'Normal' for x in anomalies]
        fig = px.scatter(ml_df, x='contracts', y='total', color='is_anomaly', hover_data=['province', 'internal_type'], title='Detección de Anomalías (Isolation Forest)', labels={'contracts': 'Contratos', 'total': 'Total USD'}, color_discrete_map={'Normal': 'blue', 'Anomalía': 'red'})
        st.plotly_chart(fig, use_container_width=True)
        anomaly_count = sum(1 for x in anomalies if x == -1)
        st.write(f"**Anomalías detectadas:** {anomaly_count} de {len(ml_df)} puntos de datos.")
    except Exception as e:
        st.error(f"Error en Detección de Anomalías: {e}")

# --- FUNCIÓN PRINCIPAL DE LA APP STREAMLIT ---
def main():
    st.set_page_config(page_title="Análisis Contratos Públicos", layout="wide")
    st.title("Análisis de Compras Públicas y Graficación de Datos")
    st.subheader("Alexander Mosquera")

    if 'data_df' not in st.session_state:
        st.session_state.data_df = pd.DataFrame()
    if 'view_title' not in st.session_state:
        st.session_state.view_title = "Seleccione una opción de análisis en la barra lateral para comenzar."

    with st.sidebar:
        st.header("Opciones de Análisis")
        if st.button("Análisis Completo (Todos los datos)", help="Carga todos los años, provincias y tipos. Puede tardar varios minutos."):
            st.session_state.view_title = "Cargando análisis completo... por favor espere."
            years_to_load = list(range(2025, 2014, -1))
            st.session_state.data_df = load_all_data(years_to_load)
            st.session_state.view_title = f"Mostrando Análisis Completo: {len(st.session_state.data_df):,} registros cargados."

        st.markdown("---")
        st.header("Análisis Filtrado")
        with st.spinner("Cargando filtros..."):
            provincias_disponibles = get_provincias_from_api()
            tipos_disponibles = get_tipos_contrato_from_api()
        year_selected = st.selectbox('Año:', ['Todos'] + list(range(2025, 2014, -1)))
        provincia_selected = st.selectbox('Provincia o Entidad:', ['Todas'] + provincias_disponibles)
        tipo_selected = st.selectbox('Tipo de Contratación:', ['Todas'] + tipos_disponibles)
        if st.button("Aplicar Filtros"):
            with st.spinner("Consultando datos filtrados..."):
                st.session_state.data_df = load_data(year_selected, provincia_selected, tipo_selected)
                st.session_state.view_title = f"Mostrando datos para: {year_selected}, {provincia_selected}, {tipo_selected}"
    
    st.info(st.session_state.view_title)
    data = st.session_state.data_df

    if not data.empty:
        st.dataframe(data.head(50), use_container_width=True, hide_index=True)
        col1, col2, col3 = st.columns(3)
        total_contratos = int(data['contracts'].sum())
        total_monto = float(data['total'].sum())
        col1.metric("Total de Registros", f"{len(data):,}")
        col2.metric("Total de Contratos", f"{total_contratos:,}")
        col3.metric("Monto Total (USD)", f"${total_monto:,.2f}")

        st.markdown("---")
        st.header("Análisis Gráfico")
        
        # Gráficas
        fig1 = create_contracts_by_month_and_type(data); 
        if fig1: st.plotly_chart(fig1, use_container_width=True)
        fig2 = create_contracts_by_type(data); 
        if fig2: st.plotly_chart(fig2, use_container_width=True)
        fig3 = create_bar_chart_totals_by_type(data); 
        if fig3: st.plotly_chart(fig3, use_container_width=True)
        fig4 = create_line_chart_monthly_amounts(data); 
        if fig4: st.plotly_chart(fig4, use_container_width=True)
        fig5 = create_pie_chart_contracts_by_type(data); 
        if fig5: st.plotly_chart(fig5, use_container_width=True)
        fig6 = create_scatter_plot_total_vs_contracts(data); 
        if fig6: st.plotly_chart(fig6, use_container_width=True)
        fig7 = create_line_chart_contracts_by_month_type(data); 
        if fig7: st.plotly_chart(fig7, use_container_width=True)
        
        st.markdown("---")
        st.header("🤖 Análisis con Machine Learning")
        
        if ML_AVAILABLE:
            st.subheader("1. Clustering K-Means")
            apply_kmeans_clustering(data)
            st.subheader("2. Predicción con Prophet")
            apply_prophet_forecasting(data)
            st.subheader("3. Clasificación con Regresión Logística")
            apply_logistic_regression(data)
            st.subheader("4. Reducción Dimensional con PCA")
            apply_pca_analysis(data)
            st.subheader("5. Detección de Anomalías con Isolation Forest")
            apply_anomaly_detection(data)
        else:
            st.warning("Para usar Machine Learning, instala las librerías: pip install scikit-learn prophet")

if __name__ == "__main__":
    main()