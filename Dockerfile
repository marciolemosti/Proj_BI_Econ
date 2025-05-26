FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos de requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código-fonte
COPY . .

# Criar diretórios necessários
RUN mkdir -p data logs assets/icones

# Expor a porta do Streamlit
EXPOSE 8501

# Comando para iniciar o dashboard
CMD ["streamlit", "run", "src/visualizacao/dashboard.py"]
