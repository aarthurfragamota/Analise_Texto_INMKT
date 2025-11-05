#!/bin/bash

echo "======================================"
echo " INSTALADOR ANALISADOR DE TEXTO"
echo "======================================"
echo

echo "1. Verificando Python..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERRO: Python não encontrado!"
    echo "Instale o Python 3.8 ou superior"
    exit 1
fi

echo
echo "2. Criando ambiente virtual..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERRO: Falha ao criar ambiente virtual"
    exit 1
fi

echo
echo "3. Ativando ambiente virtual..."
source venv/bin/activate

echo
echo "4. Atualizando pip..."
python -m pip install --upgrade pip

echo
echo "5. Instalando dependências..."
pip install -r requirements.txt

echo
echo "6. Criando pastas de cache..."
mkdir -p ~/stanza_resources
mkdir -p ~/hf_cache
mkdir -p ~/nltk_data

echo
echo "======================================"
echo " INSTALAÇÃO CONCLUÍDA COM SUCESSO!"
echo "======================================"
echo
echo "Para executar o programa:"
echo "  - Execute: ./run.sh"
echo "  - Ou no terminal:"
echo "      source venv/bin/activate"
echo "      streamlit run app.py"
echo