"""
Módulo de componentes de visualização para o dashboard Termômetro da Economia.

Este módulo implementa componentes reutilizáveis para o dashboard,
incluindo exibição de métricas, gráficos e análises.
"""

import os
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger(__name__)

class ExibidorMetricas:
    """Classe para exibição de métricas no dashboard."""
    
    def __init__(self, diretorio_icones: str = None, icon_dir: str = None):
        """
        Inicializa o componente de exibição de métricas.
        
        Args:
            diretorio_icones: Diretório onde os ícones estão armazenados
            icon_dir: Alias para diretorio_icones (compatibilidade)
        """
        self.diretorio_icones = diretorio_icones or icon_dir
        logger.info(f"ExibidorMetricas inicializado com diretório de ícones: {self.diretorio_icones}")
    
    def _carregar_icone(self, nome_icone: str) -> Optional[str]:
        """
        Carrega um ícone do diretório de ícones.
        
        Args:
            nome_icone: Nome do arquivo do ícone
            
        Returns:
            Caminho completo para o ícone ou None se não encontrado
        """
        if not self.diretorio_icones or not nome_icone:
            return None
            
        caminho_icone = os.path.join(self.diretorio_icones, nome_icone)
        if os.path.exists(caminho_icone):
            return caminho_icone
        else:
            logger.warning(f"Ícone não encontrado: {caminho_icone}")
            return None
    
    def exibir_metricas(self, dados_indicadores: Dict[str, pd.DataFrame], 
                       config_indicadores: Dict[str, Dict[str, Any]]) -> None:
        """
        Exibe métricas para os indicadores configurados.
        
        Args:
            dados_indicadores: Dicionário com DataFrames dos indicadores
            config_indicadores: Configuração dos indicadores
        """
        # Filtrar apenas indicadores com dados disponíveis
        indicadores_disponiveis = {k: v for k, v in dados_indicadores.items() if not v.empty}
        
        if not indicadores_disponiveis:
            st.warning("Nenhum dado disponível para exibição de métricas.")
            return
        
        # Criar colunas para as métricas
        colunas = st.columns(len(indicadores_disponiveis))
        
        # Exibir métricas em cada coluna
        for i, (id_indicador, df) in enumerate(indicadores_disponiveis.items()):
            if id_indicador in config_indicadores:
                config = config_indicadores[id_indicador]
                
                # Obter o último valor
                if not df.empty:
                    ultima_linha = df.iloc[-1]
                    ultima_data = ultima_linha['data']
                    
                    # Obter o valor da coluna configurada ou usar a primeira coluna numérica disponível
                    coluna_valor = config.get('coluna_valor')
                    if coluna_valor and coluna_valor in ultima_linha:
                        valor = ultima_linha[coluna_valor]
                    elif 'valor' in ultima_linha:
                        valor = ultima_linha['valor']
                    elif 'deficit' in ultima_linha:
                        valor = ultima_linha['deficit']
                    elif 'iof' in ultima_linha:
                        valor = ultima_linha['iof']
                    else:
                        # Encontrar a primeira coluna numérica que não seja 'data'
                        colunas_numericas = [col for col in df.columns if col != 'data' and pd.api.types.is_numeric_dtype(df[col])]
                        if colunas_numericas:
                            valor = ultima_linha[colunas_numericas[0]]
                        else:
                            valor = None
                    
                    # Aplicar transformação se configurada
                    if valor is not None and 'transformacao_valor' in config:
                        valor = config['transformacao_valor'](valor)
                    
                    # Formatar o valor
                    if valor is not None:
                        formato = config.get('formato', '{:.2f}')
                        valor_formatado = formato.format(valor)
                        
                        # Exibir a métrica
                        with colunas[i]:
                            st.subheader(config.get('nome', id_indicador))
                            
                            # Carregar e exibir ícone se disponível
                            icone = self._carregar_icone(config.get('icone', ''))
                            if icone:
                                st.image(icone, width=50)
                            
                            st.markdown(f"<h1 style='text-align: center;'>{valor_formatado}</h1>", unsafe_allow_html=True)
                            st.caption(f"Última atualização: {ultima_data.strftime('%d/%m/%Y')}")
                else:
                    with colunas[i]:
                        st.subheader(config.get('nome', id_indicador))
                        st.warning("Dados não disponíveis")
    
    def exibir_linha_metricas(self, config_metricas: Dict[str, Dict[str, Any]]) -> None:
        """
        Exibe uma linha de métricas com base na configuração fornecida.
        
        Args:
            config_metricas: Configuração das métricas a serem exibidas
        """
        # Criar colunas para as métricas
        colunas = st.columns(len(config_metricas))
        
        # Exibir métricas em cada coluna
        for i, (id_metrica, config) in enumerate(config_metricas.items()):
            with colunas[i]:
                st.subheader(config.get('nome', id_metrica))
                
                # Carregar e exibir ícone se disponível
                icone = self._carregar_icone(config.get('icone', ''))
                if icone:
                    st.image(icone, width=50)
                
                # Exibir valor
                valor = config.get('valor', 'N/A')
                st.markdown(f"<h1 style='text-align: center;'>{valor}</h1>", unsafe_allow_html=True)
                
                # Exibir informação adicional
                if 'info' in config:
                    st.caption(config['info'])
    
    def exibir_cabecalho_secao(self, titulo: str, nome_icone: str = None) -> None:
        """
        Exibe um cabeçalho de seção com ícone opcional.
        
        Args:
            titulo: Título da seção
            nome_icone: Nome do arquivo do ícone (opcional)
        """
        col1, col2 = st.columns([0.05, 0.95])
        
        with col1:
            if nome_icone:
                icone = self._carregar_icone(nome_icone)
                if icone:
                    st.image(icone, width=30)
        
        with col2:
            st.subheader(titulo)


class ExibidorGraficos:
    """Classe para exibição de gráficos no dashboard."""
    
    def __init__(self):
        """Inicializa o componente de exibição de gráficos."""
        logger.info("ExibidorGraficos inicializado")
    
    def exibir_graficos(self, dados_indicadores: Dict[str, pd.DataFrame], 
                      config_indicadores: Dict[str, Dict[str, Any]],
                      anos_selecionados: List[int] = None) -> None:
        """
        Exibe gráficos para os indicadores selecionados.
        
        Args:
            dados_indicadores: Dicionário com DataFrames dos indicadores
            config_indicadores: Configuração dos indicadores
            anos_selecionados: Lista de anos selecionados para filtro
        """
        # Filtrar apenas indicadores com dados disponíveis
        indicadores_disponiveis = {k: v for k, v in dados_indicadores.items() if not v.empty}
        
        if not indicadores_disponiveis:
            st.warning("Nenhum dado disponível para exibição de gráficos.")
            return
        
        # Criar seletor de indicadores
        opcoes_indicadores = [(id_indicador, config.get('nome', id_indicador)) 
                            for id_indicador, config in config_indicadores.items() 
                            if id_indicador in indicadores_disponiveis]
        
        indicador_selecionado = st.selectbox(
            "Selecione o indicador para visualização:",
            [id for id, _ in opcoes_indicadores],
            format_func=lambda x: next((nome for id, nome in opcoes_indicadores if id == x), x)
        )
        
        if indicador_selecionado and indicador_selecionado in indicadores_disponiveis:
            df = indicadores_disponiveis[indicador_selecionado].copy()
            config = config_indicadores[indicador_selecionado]
            
            # Adicionar coluna de ano para filtro
            df['ano'] = df['data'].dt.year
            
            # Filtrar por anos selecionados
            if anos_selecionados:
                df = df[df['ano'].isin(anos_selecionados)]
            
            if not df.empty:
                # Obter configurações do gráfico
                titulo_grafico = config.get('titulo_grafico', f"Evolução de {config.get('nome', indicador_selecionado)}")
                
                # Obter a coluna de valor
                coluna_valor = config.get('coluna_valor')
                if coluna_valor and coluna_valor in df.columns:
                    pass  # Usar a coluna configurada
                elif 'valor' in df.columns:
                    coluna_valor = 'valor'
                elif 'deficit' in df.columns:
                    coluna_valor = 'deficit'
                elif 'iof' in df.columns:
                    coluna_valor = 'iof'
                else:
                    # Encontrar a primeira coluna numérica que não seja 'data', 'ano'
                    colunas_numericas = [col for col in df.columns if col not in ['data', 'ano'] and pd.api.types.is_numeric_dtype(df[col])]
                    if colunas_numericas:
                        coluna_valor = colunas_numericas[0]
                    else:
                        st.error(f"Não foi possível encontrar uma coluna de valor para {config.get('nome', indicador_selecionado)}")
                        return
                
                # Obter labels para o gráfico
                labels_grafico = config.get('labels_grafico', {'data': 'Data', coluna_valor: 'Valor'})
                
                # Criar gráfico
                fig = px.line(
                    df, 
                    x='data', 
                    y=coluna_valor,
                    title=titulo_grafico,
                    labels=labels_grafico
                )
                
                # Configurar formato do eixo Y
                formato_y = config.get('formato_grafico', '.2f')
                fig.update_yaxes(tickformat=formato_y)
                
                # Exibir gráfico
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Não há dados disponíveis para {config.get('nome', indicador_selecionado)} nos anos selecionados.")
    
    def exibir_correlacao(self, dados_indicadores: Dict[str, pd.DataFrame], 
                         config_indicadores: Dict[str, Dict[str, Any]],
                         anos_selecionados: List[int] = None) -> None:
        """
        Exibe análise de correlação entre dois indicadores.
        
        Args:
            dados_indicadores: Dicionário com DataFrames dos indicadores
            config_indicadores: Configuração dos indicadores
            anos_selecionados: Lista de anos selecionados para filtro
        """
        # Filtrar apenas indicadores com dados disponíveis
        indicadores_disponiveis = {k: v for k, v in dados_indicadores.items() if not v.empty}
        
        if len(indicadores_disponiveis) < 2:
            st.warning("São necessários pelo menos dois indicadores com dados para análise de correlação.")
            return
        
        # Criar seletores para os indicadores
        opcoes_indicadores = [(id_indicador, config.get('nome', id_indicador)) 
                            for id_indicador, config in config_indicadores.items() 
                            if id_indicador in indicadores_disponiveis]
        
        col1, col2 = st.columns(2)
        
        with col1:
            indicador_selecionado1 = st.selectbox(
                "Selecione o primeiro indicador para correlação:",
                [id for id, _ in opcoes_indicadores],
                format_func=lambda x: next((nome for id, nome in opcoes_indicadores if id == x), x),
                key="corr_ind1"
            )
        
        with col2:
            # Filtrar para não permitir selecionar o mesmo indicador duas vezes
            opcoes_restantes = [(id, nome) for id, nome in opcoes_indicadores if id != indicador_selecionado1]
            indicador_selecionado2 = st.selectbox(
                "Selecione o segundo indicador para correlação:",
                [id for id, _ in opcoes_restantes],
                format_func=lambda x: next((nome for id, nome in opcoes_restantes if id == x), x),
                key="corr_ind2"
            )
        
        if indicador_selecionado1 and indicador_selecionado2:
            df1 = indicadores_disponiveis[indicador_selecionado1].copy()
            df2 = indicadores_disponiveis[indicador_selecionado2].copy()
            
            config1 = config_indicadores[indicador_selecionado1]
            config2 = config_indicadores[indicador_selecionado2]
            
            # Adicionar coluna de ano para filtro
            df1['ano'] = df1['data'].dt.year
            df2['ano'] = df2['data'].dt.year
            
            # Filtrar por anos selecionados
            if anos_selecionados:
                df1 = df1[df1['ano'].isin(anos_selecionados)]
                df2 = df2[df2['ano'].isin(anos_selecionados)]
            
            if not df1.empty and not df2.empty:
                # Mesclar os DataFrames pela data
                df_mesclado = pd.merge(df1, df2, on='data', suffixes=('_1', '_2'))
                
                if df_mesclado.empty:
                    st.warning("Não há dados com datas coincidentes para os indicadores selecionados.")
                    return
                
                # Obter as colunas de valor
                coluna_valor1 = self._encontrar_coluna_valor(df_mesclado, config1, sufixo='_1')
                coluna_valor2 = self._encontrar_coluna_valor(df_mesclado, config2, sufixo='_2')
                
                if not coluna_valor1 or not coluna_valor2:
                    st.error("Não foi possível identificar as colunas de valores para correlação.")
                    st.info("Colunas disponíveis: " + ", ".join(df_mesclado.columns))
                    return
                
                # Calcular correlação
                try:
                    correlacao = df_mesclado[coluna_valor1].corr(df_mesclado[coluna_valor2])
                    
                    # Exibir resultado
                    st.metric(
                        "Coeficiente de Correlação de Pearson", 
                        f"{correlacao:.4f}",
                        help="Varia de -1 (correlação negativa perfeita) a 1 (correlação positiva perfeita). Valores próximos de 0 indicam pouca ou nenhuma correlação."
                    )
                    
                    # Interpretar o resultado
                    if abs(correlacao) < 0.3:
                        interpretacao = "Correlação fraca"
                    elif abs(correlacao) < 0.7:
                        interpretacao = "Correlação moderada"
                    else:
                        interpretacao = "Correlação forte"
                        
                    if correlacao < 0:
                        interpretacao += " negativa"
                    elif correlacao > 0:
                        interpretacao += " positiva"
                        
                    st.info(f"Interpretação: {interpretacao}")
                    
                    # Criar gráfico de dispersão
                    fig = px.scatter(
                        df_mesclado, 
                        x=coluna_valor1, 
                        y=coluna_valor2,
                        title=f"Correlação entre {config1.get('nome', indicador_selecionado1)} e {config2.get('nome', indicador_selecionado2)}",
                        labels={
                            coluna_valor1: config1.get('nome', indicador_selecionado1),
                            coluna_valor2: config2.get('nome', indicador_selecionado2)
                        },
                        trendline="ols"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Exibir tabela de dados
                    with st.expander("Ver dados utilizados na correlação"):
                        st.dataframe(df_mesclado[['data', coluna_valor1, coluna_valor2]])
                        
                except Exception as e:
                    st.error(f"Erro ao calcular correlação: {str(e)}")
                    st.info("Detalhes das colunas disponíveis:")
                    st.write(f"Colunas disponíveis: {df_mesclado.columns.tolist()}")
                    st.write(f"Tipos de dados: {df_mesclado.dtypes}")
            else:
                st.warning("Não há dados suficientes para os indicadores selecionados nos anos filtrados.")
    
    def _encontrar_coluna_valor(self, df: pd.DataFrame, config: Dict[str, Any], sufixo: str = '') -> Optional[str]:
        """
        Encontra a coluna de valor no DataFrame com base na configuração.
        
        Args:
            df: DataFrame com os dados
            config: Configuração do indicador
            sufixo: Sufixo adicionado às colunas (ex: '_1', '_2')
            
        Returns:
            Nome da coluna de valor ou None se não encontrada
        """
        # Verificar coluna configurada
        coluna_valor = config.get('coluna_valor')
        if coluna_valor:
            coluna_com_sufixo = f"{coluna_valor}{sufixo}"
            if coluna_com_sufixo in df.columns:
                return coluna_com_sufixo
        
        # Verificar colunas comuns
        for nome_coluna in ['valor', 'deficit', 'iof']:
            coluna_com_sufixo = f"{nome_coluna}{sufixo}"
            if coluna_com_sufixo in df.columns:
                return coluna_com_sufixo
        
        # Encontrar primeira coluna numérica com o sufixo
        colunas_numericas = [col for col in df.columns 
                           if col.endswith(sufixo) and 
                           col not in [f'data{sufixo}', f'ano{sufixo}'] and 
                           pd.api.types.is_numeric_dtype(df[col])]
        
        if colunas_numericas:
            return colunas_numericas[0]
        
        return None
    
    def exibir_comparativo(self, dados_indicadores: Dict[str, pd.DataFrame],
                          config_indicadores: Dict[str, Dict[str, Any]],
                          anos_selecionados: List[int] = None) -> None:
        """
        Exibe comparativo entre múltiplos indicadores.
        
        Args:
            dados_indicadores: Dicionário com DataFrames dos indicadores
            config_indicadores: Configuração dos indicadores
            anos_selecionados: Lista de anos selecionados para filtro
        """
        # Filtrar apenas indicadores com dados disponíveis
        indicadores_disponiveis = {k: v for k, v in dados_indicadores.items() if not v.empty}
        
        if len(indicadores_disponiveis) < 1:
            st.warning("Não há indicadores disponíveis para comparação.")
            return
        
        # Criar seletor de indicadores
        opcoes_indicadores = [(id_indicador, config.get('nome', id_indicador)) 
                            for id_indicador, config in config_indicadores.items() 
                            if id_indicador in indicadores_disponiveis]
        
        indicadores_selecionados = st.multiselect(
            "Selecione os indicadores para comparar:",
            [id for id, _ in opcoes_indicadores],
            format_func=lambda x: next((nome for id, nome in opcoes_indicadores if id == x), x)
        )
        
        if not indicadores_selecionados:
            st.warning("Selecione pelo menos um indicador para o comparativo.")
            return
        
        # Preparar dados para comparação
        dados_comparativo = {}
        for id_indicador in indicadores_selecionados:
            if id_indicador in indicadores_disponiveis:
                df = indicadores_disponiveis[id_indicador].copy()
                config = config_indicadores[id_indicador]
                
                # Adicionar coluna de ano para filtro
                df['ano'] = df['data'].dt.year
                
                # Filtrar por anos selecionados
                if anos_selecionados:
                    df = df[df['ano'].isin(anos_selecionados)]
                
                if not df.empty:
                    # Obter a coluna de valor
                    coluna_valor = config.get('coluna_valor')
                    if coluna_valor and coluna_valor in df.columns:
                        pass  # Usar a coluna configurada
                    elif 'valor' in df.columns:
                        coluna_valor = 'valor'
                    elif 'deficit' in df.columns:
                        coluna_valor = 'deficit'
                    elif 'iof' in df.columns:
                        coluna_valor = 'iof'
                    else:
                        # Encontrar a primeira coluna numérica que não seja 'data', 'ano'
                        colunas_numericas = [col for col in df.columns if col not in ['data', 'ano'] and pd.api.types.is_numeric_dtype(df[col])]
                        if colunas_numericas:
                            coluna_valor = colunas_numericas[0]
                        else:
                            st.error(f"Não foi possível encontrar uma coluna de valor para {config.get('nome', id_indicador)}")
                            continue
                    
                    # Normalizar os dados para comparação
                    df_norm = df[['data', coluna_valor]].copy()
                    df_norm.rename(columns={coluna_valor: config.get('nome', id_indicador)}, inplace=True)
                    
                    dados_comparativo[id_indicador] = df_norm
        
        if not dados_comparativo:
            st.warning("Não há dados disponíveis para os indicadores selecionados nos anos filtrados.")
            return
        
        # Mesclar os DataFrames pela data
        df_final = None
        for id_indicador, df in dados_comparativo.items():
            if df_final is None:
                df_final = df
            else:
                df_final = pd.merge(df_final, df, on='data', how='outer')
        
        if df_final is None or df_final.empty:
            st.warning("Não foi possível mesclar os dados dos indicadores selecionados.")
            return
        
        # Ordenar por data
        df_final.sort_values('data', inplace=True)
        
        # Criar gráfico de comparação
        fig = go.Figure()
        
        for id_indicador in indicadores_selecionados:
            if id_indicador in config_indicadores:
                nome_indicador = config_indicadores[id_indicador].get('nome', id_indicador)
                if nome_indicador in df_final.columns:
                    fig.add_trace(go.Scatter(
                        x=df_final['data'],
                        y=df_final[nome_indicador],
                        mode='lines',
                        name=nome_indicador
                    ))
        
        fig.update_layout(
            title="Comparativo de Indicadores",
            xaxis_title="Data",
            yaxis_title="Valor",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Opção para normalizar os valores
        if st.checkbox("Normalizar valores para comparação (base 100)"):
            # Criar cópia para normalização
            df_norm = df_final.copy()
            
            # Normalizar cada coluna (exceto 'data')
            for col in df_norm.columns:
                if col != 'data':
                    # Obter o primeiro valor não nulo
                    primeiro_valor = df_norm[col].dropna().iloc[0] if not df_norm[col].dropna().empty else 1
                    if primeiro_valor != 0:
                        df_norm[col] = df_norm[col] / primeiro_valor * 100
            
            # Criar gráfico normalizado
            fig_norm = go.Figure()
            
            for id_indicador in indicadores_selecionados:
                if id_indicador in config_indicadores:
                    nome_indicador = config_indicadores[id_indicador].get('nome', id_indicador)
                    if nome_indicador in df_norm.columns:
                        fig_norm.add_trace(go.Scatter(
                            x=df_norm['data'],
                            y=df_norm[nome_indicador],
                            mode='lines',
                            name=nome_indicador
                        ))
            
            fig_norm.update_layout(
                title="Comparativo de Indicadores (Normalizado - Base 100)",
                xaxis_title="Data",
                yaxis_title="Valor (Base 100)",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_norm, use_container_width=True)
