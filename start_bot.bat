@echo off
:loop
echo Starting Fennec Bot...
python bot.py
echo Bot crashed or stopped. Restarting in 10 seconds...
timeout /t 10
goto loop
