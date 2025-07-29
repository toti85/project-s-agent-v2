@echo off
echo ==========================================
echo Git Telepítő Script - Project-S V2
echo ==========================================
echo.

echo 🔍 Git ellenőrzése...
where git >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo ✅ Git már telepítve van!
    git --version
    echo.
    echo Tovább a GitHub feltöltéshez...
    pause
    goto :upload
)

echo ❌ Git nincs telepítve
echo.
echo 📥 Git letöltése és telepítése...
echo.
echo Kérlek, kövesd ezeket a lépéseket:
echo.
echo 1. Nyisd meg a böngészőt
echo 2. Menj a https://git-scm.com/download/win oldalra
echo 3. Töltsd le a Git-et (64-bit verzió ajánlott)
echo 4. Telepítsd alapértelmezett beállításokkal
echo 5. Indítsd újra a PowerShell-t
echo 6. Futtasd újra ezt a scriptet
echo.

echo Automatikus böngésző megnyitás...
start https://git-scm.com/download/win
echo.
echo ⏳ Telepítés után nyomj ENTER-t a folytatáshoz...
pause

echo.
echo 🔍 Git újraellenőrzése...
where git >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo ✅ Git sikeresen telepítve!
    git --version
    goto :upload
) else (
    echo ❌ Git még mindig nincs telepítve
    echo Kérlek, telepítsd manuálisan és indítsd újra a PowerShell-t
    pause
    exit /b 1
)

:upload
echo.
echo ==========================================
echo 🚀 GitHub Feltöltés - Git Parancsok
echo ==========================================
echo.

echo Git konfigurálása (első használat esetén):
echo git config --global user.name "Your Name"
echo git config --global user.email "your.email@example.com"
echo.

echo GitHub feltöltés parancsok:
echo.
echo cd c:\v2_architecture\project-s-clean
echo git init
echo git remote add origin https://github.com/toti85/project-s-agent-v2.git
echo git add .
echo git commit -m "Initial commit: Project-S V2 - Advanced Autonomous AI Agent"
echo git branch -M main
echo git push -u origin main
echo.

echo ⚠️ FONTOS: Bejelentkezéshez szükség lehet GitHub token-re!
echo GitHub → Settings → Developer settings → Personal access tokens
echo.

set /p choice="Futtatjam automatikusan a Git parancsokat? (y/n): "
if /i "%choice%"=="y" goto :auto_upload
if /i "%choice%"=="yes" goto :auto_upload

echo Kézi feltöltéshez használd a fenti parancsokat.
pause
exit /b 0

:auto_upload
echo.
echo 🚀 Automatikus GitHub feltöltés indítása...
echo.

git init
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Git init sikertelen
    pause
    exit /b 1
)

git remote add origin https://github.com/toti85/project-s-agent-v2.git
git add .
git commit -m "Initial commit: Project-S V2 - Advanced Autonomous AI Agent (100%% tested)"
git branch -M main

echo.
echo 🔐 GitHub bejelentkezés szükséges...
echo Ha kéri, add meg a GitHub username-t és personal access token-t
echo.

git push -u origin main
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==========================================
    echo ✅ SIKERES GITHUB FELTÖLTÉS!
    echo ==========================================
    echo.
    echo 🔗 Repository: https://github.com/toti85/project-s-agent-v2
    echo.
    echo Következő lépések:
    echo 1. Ellenőrizd a repository-t GitHub-on
    echo 2. Adj hozzá description-t és topics-okat
    echo 3. Hozz létre release-t (v2.0.0)
    echo.
) else (
    echo.
    echo ❌ GitHub feltöltés sikertelen
    echo Lehetséges okok:
    echo - Bejelentkezési probléma
    echo - Hálózati hiba
    echo - Repository nem létezik
    echo.
    echo Próbáld meg GitHub Desktop-pal vagy manuálisan.
)

pause
