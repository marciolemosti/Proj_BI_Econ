"""
Módulo principal do dashboard Termômetro da Economia.

Este módulo implementa o dashboard interativo para visualização
de indicadores econômicos e previsões.
"""

import os
import sys
import json
import logging
import datetime
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Union, Any, Tuple

# Adicionar diretório raiz ao path para importações relativas
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_raiz = os.path.dirname(os.path.dirname(diretorio_atual))
if diretorio_raiz not in sys.path:
    sys.path.insert(0, diretorio_raiz)

# Importar módulos do projeto
from src.utils.configuracao import obter_configuracao, configurar_logging
from src.visualizacao.componentes.exibidores import ExibidorMetricas, ExibidorGraficos
from src.dados.processadores.previsao import PrevisorSeriesTemporal, processar_dados_deficit, processar_dados_iof

# Configurar logging
configurar_logging()
logger = logging.getLogger(__name__)

def carregar_dados_json(nome_arquivo: str, possiveis_diretorios: List[str]) -> List[Dict[str, Any]]:
    """
    Carrega dados de um arquivo JSON, tentando vários diretórios possíveis.
    
    Args:
        nome_arquivo: Nome do arquivo JSON a ser carregado
        possiveis_diretorios: Lista de diretórios onde procurar o arquivo
        
    Returns:
        Lista de dicionários com os dados carregados ou lista vazia em caso de erro
    """
    for diretorio in possiveis_diretorios:
        caminho_arquivo = os.path.join(diretorio, nome_arquivo)
        try:
            if os.path.exists(caminho_arquivo):
                with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                    dados = json.load(f)
                logger.info(f"Arquivo {caminho_arquivo} carregado com sucesso.")
                return dados
        except Exception as e:
            logger.warning(f"Erro ao carregar arquivo {caminho_arquivo}: {e}")
    
    logger.warning(f"Arquivo {nome_arquivo} não encontrado em nenhum diretório.")
    return []

def conectar_bd():
    """
    Tenta estabelecer conexão com o banco de dados.
    
    Returns:
        Conexão com o banco de dados ou None em caso de erro
    """
    try:
        import psycopg2
        config = obter_configuracao()
        conn = psycopg2.connect(
            host=config["bd"]["host"],
            port=config["bd"]["port"],
            database=config["bd"]["database"],
            user=config["bd"]["user"],
            password=config["bd"]["password"]
        )
        logger.info("Conexão com o banco de dados estabelecida com sucesso.")
        return conn
    except Exception as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        logger.error(f"Erro ao conectar ao banco de dados: {e}")
        return None

def carregar_dados_indicador(conn, consulta: str) -> pd.DataFrame:
    """
    Carrega dados de um indicador do banco de dados.
    
    Args:
        conn: Conexão com o banco de dados
        consulta: Consulta SQL para obter os dados
        
    Returns:
        DataFrame com os dados do indicador ou DataFrame vazio em caso de erro
    """
    try:
        df = pd.read_sql_query(consulta, conn)
        return df
    except Exception as e:
        logger.error(f"Erro ao executar consulta: {e}")
        return pd.DataFrame()

def main():
    """Função principal do dashboard."""
    # Obter configuração
    config = obter_configuracao()
    
    # Configurar página
    st.set_page_config(
        page_title=config["visualizacao"]["dashboard"]["titulo_pagina"],
        page_icon="📊",
        layout=config["visualizacao"]["dashboard"]["layout"]
    )
    
    # Título e subtítulo
    st.title(config["visualizacao"]["dashboard"]["titulo"])
    st.caption(config["visualizacao"]["dashboard"]["subtitulo"])
    
    # Exibir data e hora de carregamento
    agora = datetime.datetime.now()
    st.caption(f"Dashboard carregado em: {agora.strftime('%d/%m/%Y %H:%M:%S')} (Horário de Brasília). Dados atualizados conforme fontes originais.")
    
    # Tentar conectar ao banco de dados
    conn = conectar_bd()
    
    # Diretórios possíveis para dados
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    diretorio_projeto = os.path.dirname(os.path.dirname(diretorio_atual))
    
    possiveis_diretorios_dados = [
        config["caminhos"]["diretorio_dados"],
        os.path.join(diretorio_projeto, "data"),
        os.path.join(diretorio_atual, "..", "..", "data"),
        "/app/data"
    ]
    
    # Carregar dados dos indicadores
    dados_indicadores = {}
    
    # Tentar carregar do banco de dados primeiro
    if conn:
        try:
            for id_indicador, config_indicador in config["visualizacao"]["indicadores"].items():
                if "consulta" in config_indicador:
                    df = carregar_dados_indicador(conn, config_indicador["consulta"])
                    if not df.empty:
                        # Renomear colunas para padrão
                        if "data_referencia" in df.columns:
                            df.rename(columns={"data_referencia": "data"}, inplace=True)
                        
                        dados_indicadores[id_indicador] = df
            
            conn.close()
            logger.info("Conexão com o banco de dados fechada.")
        except Exception as e:
            logger.error(f"Erro ao carregar dados do banco de dados: {e}")
    
    # Carregar dados de déficit primário e IOF de arquivos JSON
    # Déficit Primário
    if "deficit_primario" not in dados_indicadores or dados_indicadores["deficit_primario"].empty:
        dados_deficit = carregar_dados_json("deficit_primario.json", possiveis_diretorios_dados)
        if dados_deficit:
            df_deficit = processar_dados_deficit(dados_deficit)
            if not df_deficit.empty:
                dados_indicadores["deficit_primario"] = df_deficit
    
    # Arrecadação de IOF
    if "iof" not in dados_indicadores or dados_indicadores["iof"].empty:
        dados_iof = carregar_dados_json("arrecadacao_iof.json", possiveis_diretorios_dados)
        if dados_iof:
            df_iof = processar_dados_iof(dados_iof)
            if not df_iof.empty:
                dados_indicadores["iof"] = df_iof
    
    # Diretório de assets
    assets_dir = config["caminhos"]["diretorio_assets"]
    if not os.path.exists(assets_dir):
        assets_dir = os.path.join(diretorio_projeto, "assets")
    
    # Inicializar componentes de visualização
    exibidor_metricas = ExibidorMetricas(diretorio_icones=assets_dir)
    exibidor_graficos = ExibidorGraficos()
    
    # Seção de métricas
    exibidor_metricas.exibir_cabecalho_secao(
        config["visualizacao"]["secoes"]["metricas"]["titulo"],
        config["visualizacao"]["secoes"]["metricas"]["icone"]
    )
    
    exibidor_metricas.exibir_metricas(dados_indicadores, config["visualizacao"]["indicadores"])
    
    # Seletor de anos para filtro
    anos_disponiveis = set()
    for df in dados_indicadores.values():
        if not df.empty and "data" in df.columns:
            anos = df["data"].dt.year.unique()
            anos_disponiveis.update(anos)
    
    anos_disponiveis = sorted(list(anos_disponiveis))
    
    if anos_disponiveis:
        # Selecionar os últimos 3 anos por padrão, ou todos se houver menos de 3
        default_anos = anos_disponiveis[-3:] if len(anos_disponiveis) > 3 else anos_disponiveis
        
        anos_selecionados = st.multiselect(
            "Selecione os anos para visualização:",
            anos_disponiveis,
            default=default_anos
        )
    else:
        anos_selecionados = []
    
    # Seção de gráficos
    exibidor_metricas.exibir_cabecalho_secao(
        f"{config['visualizacao']['secoes']['graficos']['titulo']} ({', '.join(map(str, anos_selecionados))})",
        config["visualizacao"]["secoes"]["graficos"]["icone"]
    )
    
    exibidor_graficos.exibir_graficos(dados_indicadores, config["visualizacao"]["indicadores"], anos_selecionados)
    
    # Seção de correlação
    exibidor_metricas.exibir_cabecalho_secao(
        f"{config['visualizacao']['secoes']['correlacao']['titulo']} ({', '.join(map(str, anos_selecionados))})",
        config["visualizacao"]["secoes"]["correlacao"]["icone"]
    )
    
    exibidor_graficos.exibir_correlacao(dados_indicadores, config["visualizacao"]["indicadores"], anos_selecionados)
    
    # Seção de previsão
    exibidor_metricas.exibir_cabecalho_secao(
        config["visualizacao"]["secoes"]["previsao"]["titulo"],
        config["visualizacao"]["secoes"]["previsao"]["icone"]
    )
    
    # Seletor de indicador para previsão
    opcoes_indicadores = [(id_indicador, config_indicador.get("nome", id_indicador)) 
                        for id_indicador, config_indicador in config["visualizacao"]["indicadores"].items() 
                        if id_indicador in dados_indicadores and not dados_indicadores[id_indicador].empty]
    
    if opcoes_indicadores:
        indicador_selecionado = st.selectbox(
            "Selecione o indicador para previsão:",
            [id for id, _ in opcoes_indicadores],
            format_func=lambda x: next((nome for id, nome in opcoes_indicadores if id == x), x)
        )
        
        # Seletor de período de previsão
        periodo_previsao = st.slider(
            "Período de previsão (meses):",
            min_value=1,
            max_value=24,
            value=12
        )
        
        # Configurações avançadas de previsão
        with st.expander("Configurações avançadas de previsão"):
            col1, col2 = st.columns(2)
            
            with col1:
                sazonalidade_anual = st.checkbox("Sazonalidade anual", value=True)
                sazonalidade_semanal = st.checkbox("Sazonalidade semanal", value=False)
                sazonalidade_diaria = st.checkbox("Sazonalidade diária", value=False)
            
            with col2:
                modo_sazonalidade = st.selectbox(
                    "Modo de sazonalidade:",
                    ["multiplicative", "additive"],
                    index=0
                )
                escala_prior = st.slider(
                    "Escala do prior para pontos de mudança:",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.05,
                    step=0.01
                )
        
        # Botão para gerar previsão
        if st.button("Gerar Previsão"):
            if indicador_selecionado in dados_indicadores:
                df = dados_indicadores[indicador_selecionado]
                config_indicador = config["visualizacao"]["indicadores"][indicador_selecionado]
                
                # Obter a coluna de valor
                coluna_valor = config_indicador.get("coluna_valor")
                if coluna_valor and coluna_valor in df.columns:
                    pass  # Usar a coluna configurada
                elif "valor" in df.columns:
                    coluna_valor = "valor"
                elif "deficit" in df.columns:
                    coluna_valor = "deficit"
                elif "iof" in df.columns:
                    coluna_valor = "iof"
                else:
                    # Encontrar a primeira coluna numérica que não seja 'data', 'ano'
                    colunas_numericas = [col for col in df.columns if col not in ["data", "ano"] and pd.api.types.is_numeric_dtype(df[col])]
                    if colunas_numericas:
                        coluna_valor = colunas_numericas[0]
                    else:
                        st.error(f"Não foi possível encontrar uma coluna de valor para {config_indicador.get('nome', indicador_selecionado)}")
                        coluna_valor = None
                
                if coluna_valor:
                    with st.spinner("Gerando previsão..."):
                        # Criar e treinar modelo
                        previsor = PrevisorSeriesTemporal(
                            sazonalidade_anual=sazonalidade_anual,
                            sazonalidade_semanal=sazonalidade_semanal,
                            sazonalidade_diaria=sazonalidade_diaria,
                            modo_sazonalidade=modo_sazonalidade,
                            escala_prior_pontos_mudanca=escala_prior
                        )
                        
                        df_preparado = previsor.preparar_dados(df, "data", coluna_valor)
                        
                        if previsor.treinar(df_preparado):
                            # Gerar previsão
                            previsao = previsor.prever(periodo_previsao)
                            
                            if previsao is not None:
                                # Criar gráficos
                                fig_previsao = previsor.plotar_previsao(
                                    previsao, 
                                    f"Previsão de {config_indicador.get('nome', indicador_selecionado)} para {periodo_previsao} meses"
                                )
                                
                                fig_componentes = previsor.plotar_componentes(previsao)
                                
                                # Exibir gráficos
                                if fig_previsao:
                                    st.plotly_chart(fig_previsao, use_container_width=True)
                                
                                if fig_componentes:
                                    st.plotly_chart(fig_componentes, use_container_width=True)
                                
                                # Exibir tabela de previsão
                                with st.expander("Ver tabela de previsão"):
                                    # Filtrar apenas as colunas relevantes
                                    colunas_exibir = ["ds", "yhat", "yhat_lower", "yhat_upper"]
                                    df_exibir = previsao[colunas_exibir].copy()
                                    
                                    # Renomear colunas para melhor compreensão
                                    df_exibir.rename(columns={
                                        "ds": "Data",
                                        "yhat": "Previsão",
                                        "yhat_lower": "Limite Inferior",
                                        "yhat_upper": "Limite Superior"
                                    }, inplace=True)
                                    
                                    # Formatar datas
                                    df_exibir["Data"] = df_exibir["Data"].dt.strftime("%d/%m/%Y")
                                    
                                    # Exibir apenas dados futuros
                                    hoje = datetime.datetime.now().strftime("%d/%m/%Y")
                                    df_futuro = df_exibir[df_exibir["Data"] >= hoje]
                                    
                                    st.dataframe(df_futuro)
                            else:
                                st.error("Erro ao gerar previsão.")
                        else:
                            st.error("Erro ao treinar modelo de previsão.")
    else:
        st.warning("Não há indicadores disponíveis para previsão.")
    
    # Seção de comparativo
    exibidor_metricas.exibir_cabecalho_secao(
        config["visualizacao"]["secoes"]["comparativo"]["titulo"],
        config["visualizacao"]["secoes"]["comparativo"]["icone"]
    )
    
    exibidor_graficos.exibir_comparativo(dados_indicadores, config["visualizacao"]["indicadores"], anos_selecionados)


if __name__ == "__main__":
    main()
