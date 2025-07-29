# Git Telepítő PowerShell Script - Project-S V2
# =============================================

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Git Telepítő Script - Project-S V2" -ForegroundColor Yellow
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Git ellenőrzése
Write-Host "🔍 Git ellenőrzése..." -ForegroundColor Blue
$gitInstalled = Get-Command git -ErrorAction SilentlyContinue
if ($gitInstalled) {
    Write-Host "✅ Git már telepítve van!" -ForegroundColor Green
    git --version
    Write-Host ""
    Write-Host "Tovább a GitHub feltöltéshez..." -ForegroundColor Green
    Read-Host "Nyomj ENTER-t a folytatáshoz"
    & "$PSScriptRoot\github_upload.ps1"
    exit
}

Write-Host "❌ Git nincs telepítve" -ForegroundColor Red
Write-Host ""

# PowerShell alapú letöltés próbálása
Write-Host "📥 Git automatikus letöltése..." -ForegroundColor Blue

try {
    # Git letöltési URL
    $gitUrl = "https://github.com/git-for-windows/git/releases/download/v2.41.0.windows.3/Git-2.41.0.3-64-bit.exe"
    $downloadPath = "$env:TEMP\Git-installer.exe"
    
    Write-Host "⏳ Git letöltése folyamatban..." -ForegroundColor Yellow
    
    # Invoke-WebRequest használata
    Invoke-WebRequest -Uri $gitUrl -OutFile $downloadPath -UseBasicParsing
    
    Write-Host "✅ Git letöltés befejezve!" -ForegroundColor Green
    Write-Host ""
    
    # Automatikus telepítés próbálása
    Write-Host "🔧 Git telepítése..." -ForegroundColor Blue
    Write-Host "⚠️ Lehet, hogy UAC jóváhagyás szükséges!" -ForegroundColor Yellow
    
    Start-Process -FilePath $downloadPath -ArgumentList "/SILENT" -Wait -Verb RunAs
    
    Write-Host "✅ Git telepítés befejezve!" -ForegroundColor Green
    Write-Host ""
    
    # Környezeti változók frissítése
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    # Git újraellenőrzése
    Write-Host "🔍 Git újraellenőrzése..." -ForegroundColor Blue
    Start-Sleep -Seconds 2
    
    $gitInstalled = Get-Command git -ErrorAction SilentlyContinue
    if ($gitInstalled) {
        Write-Host "✅ Git sikeresen telepítve!" -ForegroundColor Green
        git --version
        Write-Host ""
        
        # Átlépés GitHub feltöltésre
        Write-Host "🚀 Tovább a GitHub feltöltéshez..." -ForegroundColor Green
        Read-Host "Nyomj ENTER-t a folytatáshoz"
        & "$PSScriptRoot\github_upload.ps1"
    } else {
        Write-Host "⚠️ Git telepítés lehet, hogy újraindítást igényel" -ForegroundColor Yellow
        Write-Host "Kérlek, indítsd újra a PowerShell-t és próbáld újra" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "❌ Automatikus telepítés sikertelen: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "📋 Manuális telepítés:" -ForegroundColor Yellow
    Write-Host "1. Menj a https://git-scm.com/download/win oldalra" -ForegroundColor White
    Write-Host "2. Töltsd le a 64-bit verziót" -ForegroundColor White
    Write-Host "3. Telepítsd alapértelmezett beállításokkal" -ForegroundColor White
    Write-Host "4. Indítsd újra a PowerShell-t" -ForegroundColor White
    Write-Host ""
    
    # Böngésző megnyitása
    Write-Host "🌐 Git letöltési oldal megnyitása..." -ForegroundColor Blue
    Start-Process "https://git-scm.com/download/win"
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Read-Host "Nyomj ENTER-t a kilépéshez"
