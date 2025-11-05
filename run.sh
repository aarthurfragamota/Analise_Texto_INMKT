#!/bin/bash

echo "======================================"
echo " ANALISADOR DE TEXTO - INICIANDO"
echo "======================================"
echo

echo "Verificando ambiente virtual..."
if [ ! -d "venv" ]; then
    echo "ERRO: Ambiente virtual não encontrado!"
    echo "Execute './install.sh' primeiro"
    exit 1
fi

echo "Ativando ambiente virtual..."
source venv/bin/activate

echo "Verificando dependências..."
python -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERRO: Dependências não instaladas!"
    echo "Execute './install.sh' primeiro"
    exit 1
fi

echo
echo "Iniciando aplicação..."
echo
echo "Aguarde... O navegador abrirá automaticamente."
echo "URL: http://localhost:8501"
echo
echo "Para parar: Ctrl+C no terminal"
echo

streamlit run app.py