@echo off
echo ======================================
echo  ANALISADOR DE TEXTO - INICIANDO
echo ======================================
echo.

echo Verificando ambiente virtual...
if not exist "venv\Scripts\activate" (
    echo ERRO: Ambiente virtual não encontrado!
    echo Execute 'install.bat' primeiro
    pause
    exit /b 1
)

echo Ativando ambiente virtual...
call venv\Scripts\activate

echo Verificando dependências...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ERRO: Dependências não instaladas!
    echo Execute 'install.bat' primeiro
    pause
    exit /b 1
)

echo.
echo Iniciando aplicação...
echo.
echo Aguarde... O navegador abrirá automaticamente.
echo URL: http://localhost:8501
echo.
echo Para parar: Ctrl+C no terminal
echo.

streamlit run app.py

pause