import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, jsonify, send_from_directory
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import json

app = Flask(__name__, static_folder='static')

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'plots')
TRANSACOES_FILE = os.path.join(DATA_DIR, 'Transa__es_Completas.csv')
CAMPANHAS_FILE = os.path.join(DATA_DIR, 'campanhas.csv')

# Ensure plots directory exists
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# --- Data Loading and Preprocessing ---
def load_data():
    try:
        transacoes = pd.read_csv(TRANSACOES_FILE)
        campanhas = pd.read_csv(CAMPANHAS_FILE)
        return transacoes, campanhas
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

transacoes_df, campanhas_df = load_data()

# --- Global variables for processed data (to avoid recomputing on each request) ---
clientes_clustered_df = None
cluster_diagnostico_data = {}
preferencia_campanhas_df = None
sugestoes_campanhas_data = []
reg_model_campanhas_coef_df = None
reg_model_clientes_coef = None
clientes_alto_gasto_df = None
sugestoes_marketing_alto_gasto_data = []

def process_all_data():
    global transacoes_df, campanhas_df, clientes_clustered_df, cluster_diagnostico_data, \
           preferencia_campanhas_df, sugestoes_campanhas_data, reg_model_campanhas_coef_df, \
           reg_model_clientes_coef, clientes_alto_gasto_df, sugestoes_marketing_alto_gasto_data

    if transacoes_df is None or campanhas_df is None:
        print("Data not loaded, cannot process.")
        return

    # Etapa 3: Clusterização de Clientes (Simplified from notebook)
    clientes = transacoes_df.groupby('cliente_id').agg(
        frequencia_compras=('frequencia_compras', 'max'), # Assuming these are already aggregated per client in the CSV
        total_gasto=('total_gasto', 'max'),
        ultima_compra=('ultima_compra', 'max')
    ).reset_index()
    
    # Handle cases where aggregation might result in NaNs if source columns are not fully populated per client
    clientes = clientes.dropna(subset=['frequencia_compras', 'total_gasto', 'ultima_compra'])
    if clientes.empty:
        print("No client data available for clustering after dropna.")
        return

    scaler = StandardScaler()
    # Ensure we only scale numeric columns and they exist
    cols_to_scale = ['frequencia_compras', 'total_gasto', 'ultima_compra']
    clientes_scaled = scaler.fit_transform(clientes[cols_to_scale])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    clientes['cluster'] = kmeans.fit_predict(clientes_scaled)

    pca = PCA(n_components=2)
    clientes[['pca1', 'pca2']] = pca.fit_transform(clientes_scaled)
    clientes_clustered_df = clientes.copy()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=clientes_clustered_df, x='pca1', y='pca2', hue='cluster', palette='Set2')
    plt.title('Clusters de Clientes')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(os.path.join(PLOTS_DIR, 'clusters_clientes.png'))
    plt.close()

    # Diagnóstico dos clusters
    cluster_diagnostico_raw = clientes_clustered_df.groupby('cluster')[cols_to_scale].mean().round(2)
    temp_diagnostico = {}
    for idx, row in cluster_diagnostico_raw.iterrows():
        tipo_cliente = ""
        if row['frequencia_compras'] > 12 and row['total_gasto'] > 5000:
            tipo_cliente = "Cliente fiel e de alto valor"
        elif row['ultima_compra'] > 250:
            tipo_cliente = "Cliente inativo"
        else:
            tipo_cliente = "Cliente de valor médio e recorrência moderada"
        temp_diagnostico[f"Cluster {idx}"] = {
            "Tipo de Cliente": tipo_cliente,
            "Frequência média de compras": row['frequencia_compras'],
            "Gasto total médio": row['total_gasto'],
            "Dias desde última compra (média)": row['ultima_compra']
        }
    cluster_diagnostico_data = temp_diagnostico
    
    # Merge cluster info back to transactions
    transacoes_df_merged = pd.merge(transacoes_df, clientes_clustered_df[['cliente_id', 'cluster']], on='cliente_id', how='left')

    # Etapa 4: Conjoint Analysis (análise da preferência por campanhas)
    # Ensure 'campanha' column exists in transacoes_df_merged
    if 'campanha' in transacoes_df_merged.columns:
        preferencia_campanhas = transacoes_df_merged.groupby('campanha').agg(
            cliente_id_nunique=('cliente_id', 'nunique'),
            valor_compra_sum=('valor_compra', 'sum'),
            frequencia_compras_sum=('frequencia_compras', 'sum'),
            total_gasto_sum=('total_gasto', 'sum')
        ).reset_index()

        preferencia_campanhas = pd.merge(preferencia_campanhas, campanhas_df, left_on='campanha', right_on='nome_campanha', how='left')
        preferencia_campanhas['gasto_medio_por_cliente'] = preferencia_campanhas['total_gasto_sum'] / preferencia_campanhas['cliente_id_nunique']
        preferencia_campanhas['roi_estimado'] = preferencia_campanhas['total_gasto_sum'] / preferencia_campanhas['custo_campanha']
        preferencia_campanhas_df = preferencia_campanhas.copy()

        plt.figure(figsize=(10, 5))
        sns.barplot(data=preferencia_campanhas_df, x='campanha', y='gasto_medio_por_cliente')
        plt.title('Gasto Médio por Cliente por Campanha')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'gasto_medio_por_cliente_campanha.png'))
        plt.close()

        plt.figure(figsize=(10, 5))
        sns.barplot(data=preferencia_campanhas_df, x='campanha', y='roi_estimado')
        plt.title('ROI Estimado por Campanha')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'roi_estimado_campanha.png'))
        plt.close()

        sugestoes_campanhas_data = [
            "Priorizar campanhas com ROI elevado, como aquelas que entregaram maior retorno por real investido.",
            "Reavaliar ou reformular campanhas com ROI baixo, focando em novos formatos ou incentivos como brindes, frete grátis, etc.",
            "Investir mais em campanhas com alto gasto médio por cliente, pois indicam maior valor percebido."
        ]

        # Regressão Linear - Impacto das Campanhas
        transacoes_reg = transacoes_df_merged.merge(campanhas_df, left_on='campanha', right_on='nome_campanha', how='left')
        features_campanha = ['custo_campanha', 'alcance', 'conversao']
        # Drop rows with NaN in features or target for regression
        transacoes_reg_cleaned = transacoes_reg.dropna(subset=features_campanha + ['total_gasto'])
        if not transacoes_reg_cleaned.empty:
            X_campanha = transacoes_reg_cleaned[features_campanha]
            y_campanha = transacoes_reg_cleaned['total_gasto']
            reg_model_campanha = LinearRegression()
            reg_model_campanha.fit(X_campanha, y_campanha)
            reg_model_campanhas_coef_df = pd.DataFrame({'Variavel': features_campanha, 'Coeficiente': reg_model_campanha.coef_}).sort_values(by='Coeficiente', ascending=False)

            plt.figure(figsize=(8, 5))
            sns.barplot(data=reg_model_campanhas_coef_df, x='Variavel', y='Coeficiente')
            plt.title('Impacto das Campanhas no Total Gasto')
            plt.ylabel('Coeficiente da Regressão')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'impacto_campanhas_total_gasto.png'))
            plt.close()
        else:
            print("Not enough data for campaign regression after cleaning NaNs.")
            reg_model_campanhas_coef_df = pd.DataFrame({'Variavel': features_campanha, 'Coeficiente': [0]*len(features_campanha)})

    else:
        print("'campanha' column not found in transactions data for conjoint analysis.")
        sugestoes_campanhas_data = ["Dados de campanha não encontrados ou incompletos para análise."]
        reg_model_campanhas_coef_df = pd.DataFrame({'Variavel': ['custo_campanha', 'alcance', 'conversao'], 'Coeficiente': [0,0,0]})
        # Create empty plots if data is missing
        for plot_name in ['gasto_medio_por_cliente_campanha.png', 'roi_estimado_campanha.png', 'impacto_campanhas_total_gasto.png']:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'Dados Indisponíveis', horizontalalignment='center', verticalalignment='center')
            plt.savefig(os.path.join(PLOTS_DIR, plot_name))
            plt.close()

    # Modelo 2: Previsão com características do cliente
    df_cliente_reg = transacoes_df_merged[['idade', 'renda_mensal', 'frequencia_compras', 'total_gasto']].copy()
    df_cliente_reg = df_cliente_reg.rename(columns={'renda_mensal': 'renda_anual'}) # Assuming 'renda_mensal' should be 'renda_anual'
    df_cliente_reg = df_cliente_reg.dropna()

    if not df_cliente_reg.empty and len(df_cliente_reg) > 1:
        features_cliente = ["idade", "renda_anual", "frequencia_compras"]
        X_reg_cliente = df_cliente_reg[features_cliente]
        y_reg_cliente = df_cliente_reg["total_gasto"]
        model_cliente = LinearRegression()
        model_cliente.fit(X_reg_cliente, y_reg_cliente)
        reg_model_clientes_coef = dict(zip(features_cliente, model_cliente.coef_))

        df_cliente_reg["total_gasto_previsto"] = model_cliente.predict(X_reg_cliente)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x="total_gasto", y="total_gasto_previsto", data=df_cliente_reg, color="blue")
        plt.title("Previsão de Total Gasto vs. Real")
        plt.xlabel("Total Gasto Real")
        plt.ylabel("Total Gasto Previsto")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'previsao_total_gasto_vs_real.png'))
        plt.close()
    else:
        print("Not enough data for client regression model after cleaning NaNs.")
        reg_model_clientes_coef = {"idade": 0, "renda_anual": 0, "frequencia_compras": 0}
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Dados Indisponíveis', horizontalalignment='center', verticalalignment='center')
        plt.savefig(os.path.join(PLOTS_DIR, 'previsao_total_gasto_vs_real.png'))
        plt.close()

    # CLV (Customer Lifetime Value)
    # Using 'total_gasto' as CLV as per the simplified part of the notebook
    clientes_clv = transacoes_df_merged[['cliente_id', 'total_gasto']].drop_duplicates(subset=['cliente_id'])
    clientes_clv = clientes_clv.rename(columns={'total_gasto': 'clv'})
    clientes_clv = clientes_clv.dropna(subset=['clv'])
    
    if not clientes_clv.empty:
        thresh_clv = clientes_clv['clv'].quantile(0.75)
        clientes_clv['segmento_valor'] = clientes_clv['clv'].apply(lambda x: 'Alto Valor' if x >= thresh_clv else 'Demais')

        plt.figure(figsize=(12, 6))
        sns.histplot(data=clientes_clv, x='clv', hue='segmento_valor', bins=30, kde=True, palette='Set2')
        plt.title('Distribuição do CLV por Segmento (CLV = Total Gasto)')
        plt.xlabel('Customer Lifetime Value')
        plt.ylabel('Frequência')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'distribuicao_clv_segmento.png'))
        plt.close()
    else:
        print("Not enough data for CLV analysis.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Dados Indisponíveis', horizontalalignment='center', verticalalignment='center')
        plt.savefig(os.path.join(PLOTS_DIR, 'distribuicao_clv_segmento.png'))
        plt.close()

    # Clientes de alto gasto
    cols_expected = {
        'cliente_id': 'Cliente ID',
        'nome_produto': 'Nome Produto',
        'categoria_produto': 'Categoria Produto',
        'preco_unitario': 'Preco Unitario',
        'quantidade': 'Quantidade',
        'data_compra': 'Data Compra',
        'metodo_pagamento': 'Metodo Pagamento',
        'status_pedido': 'Status Pedido',
        'endereco_entrega': 'Endereco Entrega',
        'cidade_entrega': 'Cidade Entrega',
        'estado_entrega': 'Estado Entrega',
        'cep_entrega': 'CEP Entrega',
        'pais_entrega': 'Pais Entrega',
        'idade': 'Idade',
        'renda_mensal': 'Renda Mensal',
        'genero': 'Genero',
        'tipo_cliente': 'Tipo Cliente',
        'frequencia_compras': 'Frequencia Compras',
        'total_gasto': 'Total Gasto',
        'ultima_compra': 'Ultima Compra',
        'campanha': 'Campanha',
        'cluster': 'Cluster'
    }
    clientes_alto_gasto_df = transacoes_df_merged[transacoes_df_merged['total_gasto'] >= 60000]
    # Select and rename columns to match frontend expectations
    if not clientes_alto_gasto_df.empty:
        # Rename 'Data' to 'data_compra' if it exists
        if 'Data' in clientes_alto_gasto_df.columns:
            clientes_alto_gasto_df = clientes_alto_gasto_df.rename(columns={'Data': 'data_compra'})
        
        actual_cols = {k:v for k,v in cols_expected.items() if k in clientes_alto_gasto_df.columns}
        clientes_alto_gasto_df = clientes_alto_gasto_df[list(actual_cols.keys())].rename(columns=actual_cols)
    else:
        # Create an empty DataFrame with expected column names if no high-value customers are found
        clientes_alto_gasto_df = pd.DataFrame(columns=list(cols_expected.values()))

    sugestoes_marketing_alto_gasto_data = [
        "Oferecer experiências exclusivas e personalizadas, como eventos VIP ou convites para lançamentos de produtos.",
        "Criar um programa de fidelidade premium com recompensas e benefícios exclusivos.",
        "Enviar comunicações personalizadas com ofertas especiais e produtos exclusivos.",
        "Desenvolver um sistema de recomendação de produtos baseado no histórico de compra de cada cliente.",
        "Oferecer atendimento personalizado e exclusivo, como um gerente de contas dedicado.",
        "Investir em campanhas de marketing direcionadas aos clientes de alto valor, com foco em produtos de luxo ou serviços exclusivos."
    ]

# --- Flask Routes ---
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/diagnostico_clusters')
def api_diagnostico_clusters():
    return jsonify(cluster_diagnostico_data)

@app.route('/api/sugestoes_campanhas')
def api_sugestoes_campanhas():
    return jsonify(sugestoes_campanhas_data)

@app.route('/api/coeficientes_regressao_campanhas')
def api_coeficientes_regressao_campanhas():
    if reg_model_campanhas_coef_df is not None:
        return jsonify(reg_model_campanhas_coef_df.to_dict(orient='records'))
    return jsonify([])

@app.route('/api/coeficientes_regressao_clientes')
def api_coeficientes_regressao_clientes():
    # Format as a list of dicts or a single dict as expected by frontend
    # The JS expects an object where keys are feature names
    if reg_model_clientes_coef is not None:
         return jsonify(reg_model_clientes_coef) # Returns a dict like {'idade': coef1, ...}
    return jsonify({})

@app.route('/api/clientes_alto_gasto')
def api_clientes_alto_gasto():
    if clientes_alto_gasto_df is not None:
        # Convert to list of dicts, handle NaN as null for JSON
        return jsonify(json.loads(clientes_alto_gasto_df.to_json(orient='records', date_format='iso')))
    return jsonify([])

@app.route('/api/sugestoes_marketing_alto_gasto')
def api_sugestoes_marketing_alto_gasto():
    return jsonify(sugestoes_marketing_alto_gasto_data)

# Run initial data processing when the app starts
if __name__ == '__main__':
    # This check is important to avoid running it multiple times if using a development server with reloader
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
        process_all_data()
    app.run(host='0.0.0.0', port=5000)
else:
    # For Gunicorn or other production servers, process data when the module is imported.
    process_all_data()

