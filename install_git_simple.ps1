# Git Installer for Windows
# =========================

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Git Installer for Windows" -ForegroundColor Yellow
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Git is already installed
$gitInstalled = Get-Command git -ErrorAction SilentlyContinue
if ($gitInstalled) {
    Write-Host "Git is already installed!" -ForegroundColor Green
    git --version
    Write-Host ""
    Write-Host "You can now run upload_github.ps1" -ForegroundColor Yellow
    Read-Host "Press ENTER to exit"
    exit 0
}

Write-Host "Git not found. Starting installation..." -ForegroundColor Yellow
Write-Host ""

# Try winget first
Write-Host "Trying winget installation..." -ForegroundColor Blue
$wingetResult = & winget install --id Git.Git -e --source winget 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Git installed successfully via winget!" -ForegroundColor Green
    Write-Host "Please restart your terminal and run upload_github.ps1" -ForegroundColor Yellow
    Read-Host "Press ENTER to exit"
    exit 0
}

# Try chocolatey
Write-Host "Winget failed. Trying chocolatey..." -ForegroundColor Blue
$chocoResult = & choco install git -y 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Git installed successfully via chocolatey!" -ForegroundColor Green
    Write-Host "Please restart your terminal and run upload_github.ps1" -ForegroundColor Yellow
    Read-Host "Press ENTER to exit"
    exit 0
}

# Manual installation
Write-Host "Automatic installation failed." -ForegroundColor Red
Write-Host ""
Write-Host "Please install Git manually:" -ForegroundColor Yellow
Write-Host "1. Download Git from: https://git-scm.com/download/win" -ForegroundColor White
Write-Host "2. Run the installer with default settings" -ForegroundColor White
Write-Host "3. Restart your terminal" -ForegroundColor White
Write-Host "4. Run upload_github.ps1" -ForegroundColor White
Write-Host ""

# Open browser to Git download page
$openBrowser = Read-Host "Open Git download page in browser? (y/n)"
if ($openBrowser -eq 'y' -or $openBrowser -eq 'yes') {
    Start-Process "https://git-scm.com/download/win"
    Write-Host "Git download page opened in browser" -ForegroundColor Green
}

Write-Host ""
Write-Host "After installing Git, run: upload_github.ps1" -ForegroundColor Cyan
Read-Host "Press ENTER to exit"
