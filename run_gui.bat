@echo off
title Hateful Meme Shield - Multimodal AI Dashboard
echo ==============================================================
echo     Hateful Meme Shield: Multimodal AI Classification GUI
echo ==============================================================
echo.
cd /d "%~dp0"
echo Launching Streamlit Dashboard...
streamlit run app.py
pause
