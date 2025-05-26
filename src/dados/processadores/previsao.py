"""
Módulo de processamento para modelos de previsão de séries temporais.

Este módulo contém classes e funções para criar, treinar e avaliar modelos
de previsão para séries temporais econômicas, utilizando Prophet e outras técnicas.
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, List, Optional, Union, Any, Tuple
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
from src.utils.configuracao import obter_configuracao

# Configurar logger
logger = logging.getLogger(__name__)

class PrevisorSeriesTemporal:
    """
    Classe para previsão de séries temporais econômicas.
    
    Attributes:
        tipo_modelo (str): Tipo de modelo de previsão ('prophet', 'arima', etc.).
        modelo: Modelo de previsão treinado.
    """
    
    def __init__(self, tipo_modelo: str = 'prophet', sazonalidade_anual: bool = True, 
                 sazonalidade_semanal: bool = True, sazonalidade_diaria: bool = False,
                 modo_sazonalidade: str = 'multiplicative', escala_prior_pontos_mudanca: float = 0.05):
        """
        Inicializa o previsor de séries temporais.
        
        Args:
            tipo_modelo: Tipo de modelo de previsão ('prophet', 'arima', etc.).
            sazonalidade_anual: Se deve incluir sazonalidade anual.
            sazonalidade_semanal: Se deve incluir sazonalidade semanal.
            sazonalidade_diaria: Se deve incluir sazonalidade diária.
            modo_sazonalidade: Modo de sazonalidade ('additive' ou 'multiplicative').
            escala_prior_pontos_mudanca: Escala do prior para pontos de mudança.
        """
        self.tipo_modelo = tipo_modelo
        self.modelo = None
        self.sazonalidade_anual = sazonalidade_anual
        self.sazonalidade_semanal = sazonalidade_semanal
        self.sazonalidade_diaria = sazonalidade_diaria
        self.modo_sazonalidade = modo_sazonalidade
        self.escala_prior_pontos_mudanca = escala_prior_pontos_mudanca
        
    def preparar_dados(self, df: pd.DataFrame, coluna_data: str, coluna_valor: str) -> pd.DataFrame:
        """
        Prepara os dados para o formato exigido pelo modelo de previsão.
        
        Args:
            df: DataFrame com os dados da série temporal.
            coluna_data: Nome da coluna de data.
            coluna_valor: Nome da coluna de valor.
            
        Returns:
            DataFrame formatado para o modelo de previsão.
        """
        if df.empty:
            logger.warning("DataFrame vazio fornecido para preparação de dados.")
            return pd.DataFrame()
            
        try:
            # Cria cópia para evitar modificar o original
            df_prophet = df[[coluna_data, coluna_valor]].copy()
            
            # Renomeia colunas para o formato do Prophet
            df_prophet.rename(columns={coluna_data: "ds", coluna_valor: "y"}, inplace=True)
            
            # Garante que a coluna de data está no formato datetime
            df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
            
            # Remove valores nulos
            df_prophet = df_prophet.dropna(subset=["ds", "y"])
            
            # Ordena por data
            df_prophet = df_prophet.sort_values(by="ds")
            
            logger.info(f"Dados preparados com sucesso: {len(df_prophet)} registros válidos.")
            return df_prophet
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados para previsão: {e}")
            return pd.DataFrame()
    
    def treinar(self, df: pd.DataFrame) -> bool:
        """
        Treina o modelo de previsão com os dados fornecidos.
        
        Args:
            df: DataFrame formatado para o modelo (com colunas 'ds' e 'y').
            
        Returns:
            True se o treinamento foi bem-sucedido, False caso contrário.
        """
        if df.empty or len(df) < 2:
            logger.error("Dados insuficientes para treinar o modelo.")
            return False
            
        try:
            if self.tipo_modelo == 'prophet':
                # Configura e treina o modelo Prophet
                self.modelo = Prophet(
                    yearly_seasonality=self.sazonalidade_anual,
                    weekly_seasonality=self.sazonalidade_semanal,
                    daily_seasonality=self.sazonalidade_diaria,
                    seasonality_mode=self.modo_sazonalidade,
                    changepoint_prior_scale=self.escala_prior_pontos_mudanca
                )
                self.modelo.fit(df)
                logger.info("Modelo Prophet treinado com sucesso.")
                return True
            else:
                logger.error(f"Tipo de modelo não suportado: {self.tipo_modelo}")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao treinar modelo {self.tipo_modelo}: {e}")
            return False
    
    def prever(self, periodos: int = 365) -> Optional[pd.DataFrame]:
        """
        Gera previsões para períodos futuros.
        
        Args:
            periodos: Número de períodos futuros para prever.
            
        Returns:
            DataFrame com as previsões ou None em caso de erro.
        """
        if self.modelo is None:
            logger.error("Modelo não treinado. Execute o método treinar() primeiro.")
            return None
            
        try:
            if self.tipo_modelo == 'prophet':
                # Cria dataframe de datas futuras
                futuro = self.modelo.make_future_dataframe(periods=periodos)
                
                # Gera previsões
                previsao = self.modelo.predict(futuro)
                logger.info(f"Previsão gerada com sucesso para {periodos} períodos.")
                return previsao
            else:
                logger.error(f"Tipo de modelo não suportado: {self.tipo_modelo}")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao gerar previsão: {e}")
            return None
    
    def plotar_previsao(self, previsao: pd.DataFrame, titulo: str = "Previsão de Série Temporal") -> Optional[go.Figure]:
        """
        Cria gráfico de previsão.
        
        Args:
            previsao: DataFrame com as previsões.
            titulo: Título do gráfico.
            
        Returns:
            Objeto Figure do Plotly ou None em caso de erro.
        """
        if self.modelo is None:
            logger.error("Modelo não treinado. Execute o método treinar() primeiro.")
            return None
            
        try:
            if self.tipo_modelo == 'prophet':
                fig = plot_plotly(self.modelo, previsao)
                fig.update_layout(
                    title=titulo,
                    xaxis_title="Data",
                    yaxis_title="Valor",
                    hovermode="x unified"
                )
                return fig
            else:
                logger.error(f"Tipo de modelo não suportado: {self.tipo_modelo}")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao criar gráfico de previsão: {e}")
            return None
    
    def plotar_componentes(self, previsao: pd.DataFrame) -> Optional[go.Figure]:
        """
        Cria gráfico de componentes da previsão (tendência, sazonalidade).
        
        Args:
            previsao: DataFrame com as previsões.
            
        Returns:
            Objeto Figure do Plotly ou None em caso de erro.
        """
        if self.modelo is None:
            logger.error("Modelo não treinado. Execute o método treinar() primeiro.")
            return None
            
        try:
            if self.tipo_modelo == 'prophet':
                fig = plot_components_plotly(self.modelo, previsao)
                return fig
            else:
                logger.error(f"Tipo de modelo não suportado: {self.tipo_modelo}")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao criar gráfico de componentes: {e}")
            return None


def processar_dados_deficit(dados: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Processa dados de déficit primário do BCB.
    
    Args:
        dados: Lista de dicionários com dados do déficit primário.
        
    Returns:
        DataFrame processado.
    """
    try:
        # Converte para DataFrame
        df = pd.DataFrame(dados)
        
        # Converte tipos de dados
        df['data'] = pd.to_datetime(df['data'])
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        
        # Remove valores nulos
        df = df.dropna(subset=['valor'])
        
        # Ordena por data
        df = df.sort_values(by='data')
        
        # Renomeia coluna para clareza
        df.rename(columns={'valor': 'deficit'}, inplace=True)
        
        logger.info(f"Dados de déficit primário processados: {len(df)} registros válidos.")
        return df
    except Exception as e:
        logger.error(f"Erro ao processar dados de déficit primário: {e}")
        return pd.DataFrame()


def processar_dados_iof(dados: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Processa dados de arrecadação de IOF do BCB.
    
    Args:
        dados: Lista de dicionários com dados de arrecadação de IOF.
        
    Returns:
        DataFrame processado.
    """
    try:
        # Converte para DataFrame
        df = pd.DataFrame(dados)
        
        # Converte tipos de dados
        df['data'] = pd.to_datetime(df['data'])
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        
        # Remove valores nulos
        df = df.dropna(subset=['valor'])
        
        # Ordena por data
        df = df.sort_values(by='data')
        
        # Renomeia coluna para clareza
        df.rename(columns={'valor': 'iof'}, inplace=True)
        
        logger.info(f"Dados de arrecadação de IOF processados: {len(df)} registros válidos.")
        return df
    except Exception as e:
        logger.error(f"Erro ao processar dados de arrecadação de IOF: {e}")
        return pd.DataFrame()


# Função para uso direto via linha de comando
def executar():
    """Função principal para execução direta do script."""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Exemplo de uso
    config = obter_configuracao()
    diretorio_dados = config["caminhos"]["diretorio_dados"]
    
    # Tenta carregar dados de déficit primário
    arquivo_deficit = os.path.join(diretorio_dados, "deficit_primario.json")
    if os.path.exists(arquivo_deficit):
        with open(arquivo_deficit, 'r', encoding='utf-8') as f:
            dados_deficit = json.load(f)
            
        # Processa dados
        df_deficit = processar_dados_deficit(dados_deficit)
        
        if not df_deficit.empty and len(df_deficit) > 1:
            # Cria e treina modelo para déficit primário
            previsor_deficit = PrevisorSeriesTemporal(
                sazonalidade_anual=True,
                sazonalidade_semanal=False,  # Dados mensais não têm sazonalidade semanal
                sazonalidade_diaria=False,
                modo_sazonalidade='multiplicative',
                escala_prior_pontos_mudanca=0.05
            )
            df_preparado = previsor_deficit.preparar_dados(df_deficit, 'data', 'deficit')
            
            if previsor_deficit.treinar(df_preparado):
                # Gera previsão para 24 meses (2 anos)
                previsao = previsor_deficit.prever(24)
                
                # Cria gráficos
                fig_previsao = previsor_deficit.plotar_previsao(previsao, "Previsão do Déficit Primário")
                fig_componentes = previsor_deficit.plotar_componentes(previsao)
                
                # Salva gráficos
                if fig_previsao:
                    fig_previsao.write_html(os.path.join(diretorio_dados, "previsao_deficit_primario.html"))
                    print("Gráfico de previsão do déficit primário salvo em previsao_deficit_primario.html")
                
                if fig_componentes:
                    fig_componentes.write_html(os.path.join(diretorio_dados, "componentes_deficit_primario.html"))
                    print("Gráfico de componentes do déficit primário salvo em componentes_deficit_primario.html")
    
    # Tenta carregar dados de arrecadação de IOF
    arquivo_iof = os.path.join(diretorio_dados, "arrecadacao_iof.json")
    if os.path.exists(arquivo_iof):
        with open(arquivo_iof, 'r', encoding='utf-8') as f:
            dados_iof = json.load(f)
            
        # Processa dados
        df_iof = processar_dados_iof(dados_iof)
        
        if not df_iof.empty and len(df_iof) > 1:
            # Cria e treina modelo para arrecadação de IOF
            previsor_iof = PrevisorSeriesTemporal(
                sazonalidade_anual=True,
                sazonalidade_semanal=False,  # Dados mensais não têm sazonalidade semanal
                sazonalidade_diaria=False,
                modo_sazonalidade='multiplicative',
                escala_prior_pontos_mudanca=0.05
            )
            df_preparado = previsor_iof.preparar_dados(df_iof, 'data', 'iof')
            
            if previsor_iof.treinar(df_preparado):
                # Gera previsão para 24 meses (2 anos)
                previsao = previsor_iof.prever(24)
                
                # Cria gráficos
                fig_previsao = previsor_iof.plotar_previsao(previsao, "Previsão da Arrecadação de IOF")
                fig_componentes = previsor_iof.plotar_componentes(previsao)
                
                # Salva gráficos
                if fig_previsao:
                    fig_previsao.write_html(os.path.join(diretorio_dados, "previsao_arrecadacao_iof.html"))
                    print("Gráfico de previsão da arrecadação de IOF salvo em previsao_arrecadacao_iof.html")
                
                if fig_componentes:
                    fig_componentes.write_html(os.path.join(diretorio_dados, "componentes_arrecadacao_iof.html"))
                    print("Gráfico de componentes da arrecadação de IOF salvo em componentes_arrecadacao_iof.html")


if __name__ == "__main__":
    executar()
