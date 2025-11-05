@echo off
echo ======================================
echo  INSTALADOR ANALISADOR DE TEXTO
echo ======================================
echo.

echo 1. Verificando Python...
python --version
if errorlevel 1 (
    echo ERRO: Python não encontrado!
    echo Baixe em: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo 2. Criando ambiente virtual...
python -m venv venv
if errorlevel 1 (
    echo ERRO: Falha ao criar ambiente virtual
    pause
    exit /b 1
)

echo.
echo 3. Ativando ambiente virtual...
call venv\Scripts\activate

echo.
echo 4. Atualizando pip...
python -m pip install --upgrade pip

echo.
echo 5. Instalando dependências...
pip install -r requirements.txt

echo.
echo 6. Criando pastas de cache...
mkdir D:\stanza_resources 2>nul
mkdir D:\hf_cache 2>nul
mkdir D:\nltk_data 2>nul

echo.
echo ======================================
echo  INSTALAÇÃO CONCLUÍDA COM SUCESSO!
echo ======================================
echo.
echo Para executar o programa:
echo   - Duplo clique em 'run.bat'
echo   - Ou execute no terminal:
echo        venv\Scripts\activate
echo        streamlit run app.py
echo.
pause