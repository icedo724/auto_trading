@echo off
REM 매일 장 마감 후(15:40) 신호 산출. Windows 작업 스케줄러용.
REM   schtasks /create /tn "quant-daily" /tr "C:\auto_trading\run_daily.bat" ^
REM            /sc weekly /d MON,TUE,WED,THU,FRI /st 15:40
REM
REM 이 스크립트는 주문을 내지 않는다. 신호만 만든다.
setlocal
cd /d %~dp0

if exist STOP (
  echo STOP 파일 존재 - 실행 중단
  exit /b 0
)

if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

if not defined QUANT_CONFIG set QUANT_CONFIG=configs\experiment_kr.yaml
if not defined QUANT_BEST set QUANT_BEST=reports\kr\optimization_best.json

if not exist logs mkdir logs
if not exist reports\live mkdir reports\live

if not exist "%QUANT_BEST%" (
  echo 최적 파라미터 파일이 없습니다: %QUANT_BEST%
  echo 먼저 실행하세요:  python -m quant optimize -c %QUANT_CONFIG%
  exit /b 1
)

python -m quant signal -c "%QUANT_CONFIG%" --best-file "%QUANT_BEST%" ^
    --refresh --out reports\live >> logs\daily.log 2>&1

echo 신호 산출 완료 - logs\daily.log, reports\live\signals.csv
endlocal
