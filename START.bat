@echo off
echo ============================================================
echo  CrowdGuard v3 — Starting Server
echo ============================================================
echo.

REM Install pyngrok if not already installed
echo [1/2] Installing pyngrok for mobile tunnel...
pip install pyngrok --quiet

echo [2/2] Starting CrowdGuard server...
echo.
echo  When the server starts you will see a line like:
echo  ngrok tunnel active → https://xxxx.ngrok-free.app
echo  Open THAT URL on your phone — no same-Wi-Fi needed!
echo.
python main.py
pause
