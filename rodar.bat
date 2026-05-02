@echo off
title Face Analyzer AI
color 0A

:: Vai para a pasta onde este .bat esta localizado
cd /d "%~dp0"

:: Verifica se Python esta instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  [ERRO] Python nao encontrado.
    echo  Execute instalar.bat primeiro.
    echo.
    pause
    exit /b 1
)

:: Verifica se as dependencias estao instaladas
python -c "import cv2, mediapipe, numpy, PIL" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  [AVISO] Dependencias nao instaladas.
    echo  Executando instalacao automatica...
    echo.
    call instalar.bat
)

:: Inicia o programa
echo  Iniciando Face Analyzer AI...
python main.py

:: Se der erro, mantem o terminal aberto
if %errorlevel% neq 0 (
    echo.
    echo  [ERRO] O programa encerrou com erro codigo %errorlevel%.
    echo  Veja a mensagem acima para mais detalhes.
    echo.
    pause
)
