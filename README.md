# Termômetro da Economia Brasileira

Dashboard interativo com indicadores econômicos chave do Brasil, incluindo previsões de déficit primário e arrecadação de IOF.

![Banner do Projeto](assets/banner.png)

## Sobre o Projeto

O Termômetro da Economia Brasileira é um dashboard interativo que apresenta indicadores econômicos chave do Brasil, com foco especial em déficit primário e arrecadação de IOF. O sistema extrai dados automaticamente do Banco Central do Brasil (BCB), processa séries temporais e gera previsões utilizando modelos estatísticos avançados.

### Principais Funcionalidades

- **Extração Automática de Dados**: Coleta diária de indicadores do Banco Central do Brasil
- **Previsão de Séries Temporais**: Modelos Prophet para previsão de até 24 meses
- **Dashboard Interativo**: Visualização dinâmica de indicadores e previsões
- **Análise de Correlação**: Estudo da relação entre diferentes indicadores
- **Comparativo de Indicadores**: Visualização conjunta de múltiplos indicadores

## Estrutura do Projeto

```
termometro-economia/
├── .github/workflows/      # Workflows de automação
├── assets/                 # Recursos estáticos (imagens, ícones)
├── data/                   # Dados extraídos e processados
├── docs/                   # Documentação adicional
├── src/                    # Código-fonte
│   ├── dados/              # Módulos de extração e processamento
│   │   ├── extratores/     # Extratores de dados (BCB, IBGE)
│   │   └── processadores/  # Processamento e previsão
│   ├── utils/              # Utilitários e configurações
│   └── visualizacao/       # Interface do usuário
│       └── componentes/    # Componentes reutilizáveis
├── .env.exemplo            # Exemplo de variáveis de ambiente
├── docker-compose.yml      # Configuração Docker
├── Dockerfile              # Configuração da imagem
└── requirements.txt        # Dependências
```

## Requisitos

- Python 3.11+
- Docker e Docker Compose (para execução em contêineres)
- Acesso à internet (para extração de dados do BCB)

## Instalação e Execução

### Usando Docker (Recomendado)

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/termometro-economia.git
   cd termometro-economia
   ```

2. Configure as variáveis de ambiente:
   ```bash
   cp .env.exemplo .env
   # Edite o arquivo .env conforme necessário
   ```

3. Inicie os contêineres:
   ```bash
   docker-compose up -d
   ```

4. Acesse o dashboard:
   ```
   http://localhost:8501
   ```

### Instalação Local

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/termometro-economia.git
   cd termometro-economia
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure as variáveis de ambiente:
   ```bash
   cp .env.exemplo .env
   # Edite o arquivo .env conforme necessário
   ```

5. Execute o dashboard:
   ```bash
   streamlit run src/visualizacao/dashboard.py
   ```

## Atualização Automática de Dados

O projeto está configurado para atualizar automaticamente os dados todos os dias à meia-noite (UTC) utilizando GitHub Actions. O workflow executa:

1. Extração de dados do BCB
2. Processamento e geração de previsões
3. Commit e push das alterações para o repositório

Para executar a atualização manualmente, você pode:

1. Acessar a aba "Actions" no GitHub
2. Selecionar o workflow "Atualização Automática de Dados"
3. Clicar em "Run workflow"

## Desenvolvimento

### Estrutura de Módulos

- **dados/extratores**: Contém classes e funções para extrair dados de fontes externas
- **dados/processadores**: Implementa o processamento e previsão de séries temporais
- **utils**: Utilitários como configuração e logging
- **visualizacao/componentes**: Componentes reutilizáveis para o dashboard

### Adicionando Novos Indicadores

Para adicionar um novo indicador:

1. Identifique o código da série no BCB (Sistema Gerenciador de Séries Temporais)
2. Adicione o código e nome da série em `src/utils/configuracao.py` na seção `CONFIGURACAO_EXTRACAO`
3. Configure a visualização do indicador na seção `CONFIGURACAO_VISUALIZACAO`

## Contribuição

Contribuições são bem-vindas! Para contribuir:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

## Contato

Para dúvidas ou sugestões, entre em contato através das issues do GitHub.
