@echo off
title Face Analyzer AI - Instalador
color 0A
echo.
echo  ================================================
echo   FACE ANALYZER AI - Instalador
echo  ================================================
echo.

:: Verifica se Python esta instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [!] Python nao encontrado no sistema.
    echo.
    echo  Abrindo pagina de download do Python...
    echo  Instale a versao 3.10 ou superior.
    echo  IMPORTANTE: marque "Add Python to PATH" durante a instalacao!
    echo.
    start https://www.python.org/downloads/
    echo  Apos instalar o Python, feche esta janela e rode instalar.bat novamente.
    echo.
    pause
    exit /b 1
)

echo  [OK] Python encontrado:
python --version
echo.

:: Atualiza pip
echo  [1/3] Atualizando pip...
python -m pip install --upgrade pip --quiet
echo  [OK] pip atualizado.
echo.

:: Instala dependencias
echo  [2/3] Instalando dependencias (pode demorar alguns minutos)...
echo        opencv-python, mediapipe, numpy, Pillow
echo.
pip install opencv-python mediapipe numpy Pillow
if %errorlevel% neq 0 (
    echo.
    echo  [ERRO] Falha ao instalar dependencias.
    echo  Tente rodar como Administrador.
    pause
    exit /b 1
)
echo.
echo  [OK] Dependencias instaladas.
echo.

:: Baixa o modelo do MediaPipe se nao existir
echo  [3/3] Verificando modelo MediaPipe...
if not exist "face_landmarker.task" (
    echo  Baixando modelo face_landmarker.task (~6 MB)...
    powershell -Command "Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' -OutFile 'face_landmarker.task'"
    if %errorlevel% neq 0 (
        echo  [AVISO] Nao foi possivel baixar o modelo agora.
        echo  Ele sera baixado automaticamente na primeira execucao.
    ) else (
        echo  [OK] Modelo baixado com sucesso.
    )
) else (
    echo  [OK] Modelo ja existe.
)
echo.

echo  ================================================
echo   Instalacao concluida com sucesso!
echo   Execute o arquivo rodar.bat para iniciar.
echo  ================================================
echo.
pause
