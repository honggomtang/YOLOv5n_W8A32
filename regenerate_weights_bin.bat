@echo off
REM weights.bin 재생성 (4바이트 정렬 패딩 포함, RISC-V 보드용)
REM Python 3 + PyTorch + numpy 필요. pip은 반드시 Python 3용으로: py -3 -m pip install torch numpy
setlocal
cd /d "%~dp0"

set "PT=assets/yolov5n.pt"
set "OUT=assets/weights.bin"

if not exist "%PT%" (
    echo [ERROR] %PT% not found.
    exit /b 1
)

echo Regenerating weights.bin from %PT% ...
echo.

REM YOLOv5n (anchor-based): --classic. 121 tensors + 4바이트 패딩.
REM 필수: py -3.11 -m pip install ultralytics torch numpy pandas
echo Trying Python 3.11 ...
py -3.11 --version 2>nul
py -3.11 tools/export_weights_to_bin.py --pt "%PT%" --out "%OUT%" --classic
if not errorlevel 1 goto ok
echo.
echo Trying Python 3.12 ...
py -3.12 tools/export_weights_to_bin.py --pt "%PT%" --out "%OUT%" --classic
if not errorlevel 1 goto ok
echo.
echo Trying default Python (py -3) ...
py -3 tools/export_weights_to_bin.py --pt "%PT%" --out "%OUT%" --classic
if not errorlevel 1 goto ok

echo.
echo [ERROR] Export failed. See the error message above.
echo For correct 121-tensor YOLOv5n: use Python 3.11 or 3.12, then pip install ultralytics torch numpy
exit /b 1

:ok
echo.
echo Done. %OUT% updated.
exit /b 0
