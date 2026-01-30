@echo off
REM 호스트 빌드 -> main.exe 실행 -> 골든 비교 (CMD에서 프로젝트 폴더로 cd 한 뒤 실행)
setlocal
cd /d "%~dp0"

set "GCC=C:\msys64\ucrt64\bin\gcc.exe"
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
if not exist "%GCC%" (
    echo [ERROR] gcc not found at %GCC%
    echo Add MSYS2 ucrt64\bin to PATH, or edit GCC path in this script.
    exit /b 1
)

echo [1/3] Building main.exe ...
"%GCC%" -o main.exe csrc/main.c csrc/blocks/conv.c csrc/blocks/c3.c csrc/blocks/decode.c csrc/blocks/detect.c csrc/blocks/nms.c csrc/blocks/sppf.c csrc/operations/bottleneck.c csrc/operations/concat.c csrc/operations/conv2d.c csrc/operations/maxpool2d.c csrc/operations/silu.c csrc/operations/upsample.c csrc/utils/feature_pool.c csrc/utils/image_loader.c csrc/utils/weights_loader.c csrc/utils/uart_dump.c -I. -Icsrc -std=c99 -O2 -lm
if errorlevel 1 (
    echo [ERROR] Build failed. Fix errors above, then run again.
    exit /b 1
)
echo Build OK.

echo.
echo [2/3] Running main.exe ...
main.exe
if errorlevel 1 (
    echo [ERROR] main.exe failed.
    exit /b 1
)

echo.
echo [3/3] Comparing with golden (decode_detections.py --compare) ...
python tools/decode_detections.py --compare
echo.
echo Done.
exit /b 0
