"""
Módulo de logging para o Termômetro da Economia.

Este módulo configura o sistema de logging para o projeto.
"""

import logging
from src.utils.configuracao import configurar_logging

# Configurar o sistema de logging
configurar_logging()

# Obter o logger para este módulo
logger = logging.getLogger(__name__)
