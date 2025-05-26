"""
Módulo para extração de dados do Banco Central do Brasil (BCB).

Este módulo contém classes e funções para extrair dados da API do BCB,
especificamente das séries temporais do Sistema Gerenciador de Séries Temporais (SGS).
"""

import requests
import json
import datetime
import os
import logging
from typing import Dict, List, Optional, Union, Any
from src.utils.configuracao import obter_configuracao

# Configurar logger
logger = logging.getLogger(__name__)

class ExtratorBCB:
    """
    Classe para extrair dados da API do Banco Central do Brasil.
    
    Attributes:
        url_base (str): URL base da API do BCB.
        diretorio_saida (str): Diretório para salvar os arquivos de dados.
        series (Dict[str, str]): Dicionário com códigos e nomes das séries.
    """
    
    def __init__(self, diretorio_saida: str = None):
        """
        Inicializa o extrator de dados do BCB.
        
        Args:
            diretorio_saida: Diretório para salvar os arquivos de dados.
        """
        config = obter_configuracao()
        self.url_base = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados"
        self.diretorio_saida = diretorio_saida or config["caminhos"]["diretorio_dados"]
        
        # Séries padrão para extração
        self.series = config["extracao"]["bcb"]["series"]
        
        # Criar diretório de saída se não existir
        os.makedirs(self.diretorio_saida, exist_ok=True)
        
    def buscar_dados_serie(self, codigo_serie: str, data_inicio: str, data_fim: str) -> Optional[List[Dict[str, Any]]]:
        """
        Busca dados de uma série específica da API do BCB.
        
        Args:
            codigo_serie: Código da série no SGS.
            data_inicio: Data inicial no formato DD/MM/YYYY.
            data_fim: Data final no formato DD/MM/YYYY.
            
        Returns:
            Lista de dicionários com os dados da série ou None em caso de erro.
        """
        url = f"{self.url_base.format(codigo_serie)}?formato=json&dataInicial={data_inicio}&dataFinal={data_fim}"
        
        try:
            logger.info(f"Buscando dados para série {codigo_serie} de {data_inicio} até {data_fim}")
            resposta = requests.get(url, timeout=60)
            resposta.raise_for_status()  # Lança exceção para erros HTTP
            return resposta.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao buscar dados para a série {codigo_serie}: {e}")
            return None
    
    def extrair_todas_series(self, dias_retroativos: int = None) -> Dict[str, bool]:
        """
        Extrai dados de todas as séries configuradas.
        
        Args:
            dias_retroativos: Número de dias para buscar dados retroativamente.
            
        Returns:
            Dicionário com o status de extração de cada série.
        """
        config = obter_configuracao()
        if dias_retroativos is None:
            dias_retroativos = config["extracao"]["bcb"]["dias_retroativos"]
            
        # Define a data de hoje e a data de dias_retroativos atrás
        data_fim = datetime.date.today()
        data_inicio = data_fim - datetime.timedelta(days=dias_retroativos)
        
        # Formata as datas para o formato da API (dd/MM/yyyy)
        data_inicio_str = data_inicio.strftime('%d/%m/%Y')
        data_fim_str = data_fim.strftime('%d/%m/%Y')
        
        resultados = {}
        
        # Busca e salva os dados para cada série
        for codigo, nome in self.series.items():
            logger.info(f"Iniciando extração de dados para {nome} (SGS {codigo})...")
            dados = self.buscar_dados_serie(codigo, data_inicio_str, data_fim_str)
            
            if dados:
                # Processar os dados para formato padronizado
                dados_processados = self._processar_dados(dados, nome)
                
                caminho_arquivo = os.path.join(self.diretorio_saida, f"{nome}.json")
                try:
                    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
                        json.dump(dados_processados, f, ensure_ascii=False, indent=4)
                    logger.info(f"Dados de {nome} salvos em {caminho_arquivo}")
                    resultados[nome] = True
                except IOError as e:
                    logger.error(f"Erro ao salvar o arquivo {caminho_arquivo}: {e}")
                    resultados[nome] = False
            else:
                logger.warning(f"Não foi possível obter dados para {nome}.")
                resultados[nome] = False
        
        logger.info("Coleta de dados do BCB concluída.")
        return resultados
    
    def _processar_dados(self, dados: List[Dict[str, Any]], nome_serie: str) -> List[Dict[str, Any]]:
        """
        Processa os dados da série para um formato padronizado.
        
        Args:
            dados: Lista de dicionários com os dados da série.
            nome_serie: Nome da série para identificação.
            
        Returns:
            Lista de dicionários com os dados processados.
        """
        dados_processados = []
        
        for item in dados:
            # Converter data para formato ISO
            try:
                data_str = item.get('data')
                data_obj = datetime.datetime.strptime(data_str, '%d/%m/%Y')
                data_iso = data_obj.strftime('%Y-%m-%d')
                
                # Converter valor para float
                valor_str = item.get('valor', '0').replace(',', '.')
                valor = float(valor_str)
                
                # Criar item processado
                item_processado = {
                    'data': data_iso,
                    'valor': valor
                }
                
                dados_processados.append(item_processado)
            except (ValueError, KeyError) as e:
                logger.warning(f"Erro ao processar item da série {nome_serie}: {e}")
                continue
        
        return dados_processados
    
    def adicionar_serie(self, codigo: str, nome: str) -> None:
        """
        Adiciona uma nova série para extração.
        
        Args:
            codigo: Código da série no SGS.
            nome: Nome para identificação da série.
        """
        self.series[codigo] = nome
        logger.info(f"Série {nome} (código {codigo}) adicionada para extração.")
    
    def remover_serie(self, codigo: str) -> bool:
        """
        Remove uma série da lista de extração.
        
        Args:
            codigo: Código da série no SGS.
            
        Returns:
            True se a série foi removida, False se não existia.
        """
        if codigo in self.series:
            nome = self.series[codigo]
            del self.series[codigo]
            logger.info(f"Série {nome} (código {codigo}) removida da extração.")
            return True
        return False


# Função para uso direto via linha de comando
def executar():
    """Função principal para execução direta do script."""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Criar extrator e executar
    extrator = ExtratorBCB()
    resultados = extrator.extrair_todas_series()
    
    # Exibir resultados
    for nome, sucesso in resultados.items():
        status = "sucesso" if sucesso else "falha"
        print(f"Extração de {nome}: {status}")


if __name__ == "__main__":
    executar()
