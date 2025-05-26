"""
Módulo de configuração centralizada para o Termômetro da Economia.

Este módulo centraliza todas as configurações do projeto, incluindo:
- Credenciais de banco de dados
- Parâmetros de extração de dados
- Configurações de logging
- Caminhos de diretórios
"""

import os
import base64
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração de diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Criar diretórios se não existirem
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(os.path.join(ASSETS_DIR, "icones"), exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def decodificar_base64(texto_codificado: str) -> str:
    """Decodifica uma string codificada em Base64."""
    return base64.b64decode(texto_codificado).decode('utf-8')

def codificar_base64(texto: str) -> str:
    """Codifica uma string em Base64."""
    return base64.b64encode(texto.encode('utf-8')).decode('utf-8')

# Configuração do banco de dados
# Em produção, estas credenciais devem ser obtidas de variáveis de ambiente ou secrets
CONFIGURACAO_BD = {
    "host": os.environ.get("DB_HOST", "postgres"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "database": os.environ.get("DB_NAME", "economia"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "projetobi_123"),
    "pool_mode": os.environ.get("DB_POOL_MODE", "transaction")
}

# Configuração de extração de dados
CONFIGURACAO_EXTRACAO = {
    # BCB - Banco Central do Brasil
    "bcb": {
        "series": {
            "11": "selic",
            "433": "ipca",
            "1": "cambio_ptax_venda",
            "7414": "arrecadacao_iof",
            "7415": "deficit_primario"
        },
        "dias_retroativos": 5 * 365  # 5 anos
    },
    # IBGE
    "ibge": {
        "agregados": {
            "desemprego": {
                "id": 6381,
                "variavel": 4099,
                "nivel_geografico": "n1",
                "localidade": "1"
            }
        }
    }
}

# Configuração de logging
CONFIGURACAO_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "padrao": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "padrao",
            "stream": "ext://sys.stdout"
        },
        "arquivo": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "padrao",
            "filename": os.path.join(LOGS_DIR, "termometro_economia.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "arquivo"],
            "level": "INFO",
            "propagate": True
        }
    }
}

# Configuração de visualização
CONFIGURACAO_VISUALIZACAO = {
    "dashboard": {
        "titulo": "🇧🇷 Termômetro da Economia Brasileira",
        "subtitulo": "Dashboard interativo com indicadores econômicos chave do Brasil.",
        "titulo_pagina": "Termômetro da Economia",
        "layout": "wide",
        "caminho_banner": os.path.join(ASSETS_DIR, "banner.png"),
        "largura_banner": 600
    },
    "indicadores": {
        "selic": {
            "nome": "Selic",
            "icone": "selic.png",
            "consulta": "SELECT data_referencia, taxa_selic_percentual AS selic FROM public.stg_selic ORDER BY data_referencia ASC;",
            "coluna_valor": "selic",
            "rotulo": "Selic (% a.a.)",
            "formato": "{:.2f}%",
            "titulo_grafico": "Taxa Selic (% a.a.)",
            "rotulos_grafico": {"data_referencia": "Data", "selic": "Taxa (%)"},
            "formato_grafico": ".2f%"
        },
        "ipca": {
            "nome": "IPCA",
            "icone": "inflacao.png",
            "consulta": "SELECT data_referencia, indice_ipca AS ipca FROM public.stg_ipca ORDER BY data_referencia ASC;",
            "coluna_valor": "ipca",
            "rotulo": "IPCA (Índice)",
            "formato": "{:.2f}",
            "titulo_grafico": "IPCA (Índice)",
            "rotulos_grafico": {"data_referencia": "Data", "ipca": "Índice"},
            "formato_grafico": ".2f"
        },
        "cambio": {
            "nome": "Câmbio",
            "icone": "cambio.png",
            "consulta": "SELECT data_referencia, cambio_ptax_venda_brl_usd AS cambio FROM public.stg_cambio_ptax_venda ORDER BY data_referencia ASC;",
            "coluna_valor": "cambio",
            "rotulo": "Câmbio (R$/US$)",
            "formato": "R$ {:.2f}",
            "titulo_grafico": "Câmbio (R$/US$ - PTAX Venda)",
            "rotulos_grafico": {"data_referencia": "Data", "cambio": "Taxa (R$/US$)"},
            "formato_grafico": ".2f",
            "forcar_atualizacao_data": True
        },
        "desemprego": {
            "nome": "Desemprego",
            "icone": "desemprego.png",
            "consulta": "SELECT data_referencia, taxa_desemprego_percentual AS desemprego FROM public.stg_desemprego ORDER BY data_referencia ASC;",
            "coluna_valor": "desemprego",
            "rotulo": "Desemprego (%)",
            "formato": "{:.1f}%",
            "titulo_grafico": "Taxa de Desocupação (% - PNAD Contínua)",
            "rotulos_grafico": {"data_referencia": "Data", "desemprego": "Taxa (%)"},
            "formato_grafico": ".1f%"
        },
        "pib": {
            "nome": "PIB",
            "icone": "pib.png",
            "consulta": "SELECT data_referencia, pib_valor_corrente_brl_milhoes AS pib FROM public.stg_pib_trimestral ORDER BY data_referencia ASC;",
            "coluna_valor": "pib",
            "rotulo": "PIB (R$ Bilhões)",
            "formato": "R$ {:.2f} Bi",
            "transformacao_valor": lambda x: x / 1e3,  # Converter de milhões para bilhões
            "titulo_grafico": "PIB Trimestral (R$ Milhões)",
            "rotulos_grafico": {"data_referencia": "Data", "pib": "Valor (R$ Milhões)"},
            "formato_grafico": ",.0f"
        },
        "deficit_primario": {
            "nome": "Déficit Primário",
            "icone": "deficit.png",
            "coluna_valor": "deficit",
            "rotulo": "Déficit Primário (R$ bi)",
            "formato": "{:.2f} bi",
            "titulo_grafico": "Déficit Primário (R$ bilhões)",
            "rotulos_grafico": {"data": "Data", "deficit": "Valor (R$ bilhões)"},
            "formato_grafico": ",.2f"
        },
        "iof": {
            "nome": "Arrecadação IOF",
            "icone": "iof.png",
            "coluna_valor": "iof",
            "rotulo": "Arrecadação IOF (R$ bi)",
            "formato": "{:.2f} bi",
            "titulo_grafico": "Arrecadação de IOF (R$ bilhões)",
            "rotulos_grafico": {"data": "Data", "iof": "Valor (R$ bilhões)"},
            "formato_grafico": ",.2f"
        }
    },
    "secoes": {
        "metricas": {
            "titulo": "Últimos Valores Registrados",
            "icone": "meta.png"
        },
        "graficos": {
            "titulo": "Visualização Histórica de Indicadores Macroeconômicos",
            "icone": "mercado-de-acoes.png"
        },
        "correlacao": {
            "titulo": "Análise de Correlação",
            "icone": "dispersao-espalhar.png"
        },
        "previsao": {
            "titulo": "Previsão de Indicadores",
            "icone": "previsao.png"
        },
        "comparativo": {
            "titulo": "Comparativo de Indicadores",
            "icone": "chart-bar"
        }
    }
}

# Inicialização do logging
def configurar_logging():
    """Configura o sistema de logging conforme as definições em CONFIGURACAO_LOGGING."""
    import logging.config
    logging.config.dictConfig(CONFIGURACAO_LOGGING)
    logging.info("Sistema de logging inicializado.")

# Função para obter configuração completa
def obter_configuracao() -> Dict[str, Any]:
    """Retorna a configuração completa do projeto."""
    return {
        "bd": CONFIGURACAO_BD,
        "extracao": CONFIGURACAO_EXTRACAO,
        "logging": CONFIGURACAO_LOGGING,
        "visualizacao": CONFIGURACAO_VISUALIZACAO,
        "caminhos": {
            "diretorio_base": BASE_DIR,
            "diretorio_dados": DATA_DIR,
            "diretorio_assets": ASSETS_DIR,
            "diretorio_logs": LOGS_DIR
        }
    }
