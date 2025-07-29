# GitHub Feltöltő PowerShell Script - Project-S V2
# ================================================

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "GitHub Feltöltő Script - Project-S V2" -ForegroundColor Yellow
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Git ellenőrzése
$gitInstalled = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitInstalled) {
    Write-Host "❌ Git nincs telepítve!" -ForegroundColor Red
    Write-Host "Kérlek, futtasd előbb az install_git.ps1 scriptet" -ForegroundColor Yellow
    Read-Host "Nyomj ENTER-t a kilépéshez"
    exit 1
}

Write-Host "✅ Git megtalálva:" -ForegroundColor Green
git --version
Write-Host ""

# Aktuális könyvtár ellenőrzése
$currentPath = Get-Location
Write-Host "📁 Aktuális könyvtár: $currentPath" -ForegroundColor Blue

# Project-S V2 fájlok ellenőrzése
$requiredFiles = @("README.md", "main.py", "complex_task_tester.py", "requirements.txt", "LICENSE")
$missingFiles = @()

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "❌ Hiányzó fájlok: $($missingFiles -join ', ')" -ForegroundColor Red
    Write-Host "Kérlek, navigálj a c:\v2_architecture\project-s-clean könyvtárba" -ForegroundColor Yellow
    Read-Host "Nyomj ENTER-t a kilépéshez"
    exit 1
}

Write-Host "✅ Minden szükséges fájl jelen van" -ForegroundColor Green
Write-Host ""

# Git konfiguráció ellenőrzése
Write-Host "🔧 Git konfiguráció ellenőrzése..." -ForegroundColor Blue

$gitUser = git config --global user.name
$gitEmail = git config --global user.email

if (-not $gitUser -or -not $gitEmail) {
    Write-Host "⚠️ Git konfiguráció hiányzik" -ForegroundColor Yellow
    Write-Host ""
    
    if (-not $gitUser) {
        $userName = Read-Host "Add meg a GitHub felhasználóneved"
        git config --global user.name "$userName"
        Write-Host "✅ Felhasználónév beállítva: $userName" -ForegroundColor Green
    }
    
    if (-not $gitEmail) {
        $userEmail = Read-Host "Add meg a GitHub email címed"
        git config --global user.email "$userEmail"
        Write-Host "✅ Email cím beállítva: $userEmail" -ForegroundColor Green
    }
    Write-Host ""
} else {
    Write-Host "✅ Git konfiguráció rendben:" -ForegroundColor Green
    Write-Host "   Név: $gitUser" -ForegroundColor White
    Write-Host "   Email: $gitEmail" -ForegroundColor White
    Write-Host ""
}

# GitHub feltöltés megerősítése
Write-Host "🚀 Kész a GitHub feltöltésre!" -ForegroundColor Green
Write-Host ""
Write-Host "Repository: https://github.com/toti85/project-s-agent-v2" -ForegroundColor Cyan
Write-Host ""

$confirm = Read-Host "Feltöltés most? (y/n)"
if ($confirm -ne 'y' -and $confirm -ne 'yes') {
    Write-Host "Feltöltés megszakítva" -ForegroundColor Yellow
    Read-Host "Nyomj ENTER-t a kilépéshez"
    exit 0
}

Write-Host ""
Write-Host "🔄 GitHub feltöltés folyamatban..." -ForegroundColor Blue
Write-Host ""

try {
    # Git repository inicializálása
    Write-Host "1️⃣ Git repository inicializálása..." -ForegroundColor Yellow
    git init
    if ($LASTEXITCODE -ne 0) { throw "Git init sikertelen" }
    Write-Host "✅ Git init befejezve" -ForegroundColor Green
    
    # Remote repository hozzáadása
    Write-Host "2️⃣ Remote repository hozzáadása..." -ForegroundColor Yellow
    git remote add origin https://github.com/toti85/project-s-agent-v2.git 2>$null
    Write-Host "✅ Remote origin beállítva" -ForegroundColor Green
    
    # Fájlok hozzáadása
    Write-Host "3️⃣ Fájlok hozzáadása..." -ForegroundColor Yellow
    git add .
    if ($LASTEXITCODE -ne 0) { throw "Git add sikertelen" }
    Write-Host "✅ Minden fájl hozzáadva" -ForegroundColor Green
    
    # Commit létrehozása
    Write-Host "4️⃣ Commit létrehozása..." -ForegroundColor Yellow
    git commit -m "Initial commit: Project-S V2 - Advanced Autonomous AI Agent (100% tested)"
    if ($LASTEXITCODE -ne 0) { throw "Git commit sikertelen" }
    Write-Host "✅ Commit létrehozva" -ForegroundColor Green
    
    # Main branch beállítása
    Write-Host "5️⃣ Main branch beállítása..." -ForegroundColor Yellow
    git branch -M main
    Write-Host "✅ Main branch beállítva" -ForegroundColor Green
    
    # Push GitHub-ra
    Write-Host "6️⃣ Feltöltés GitHub-ra..." -ForegroundColor Yellow
    Write-Host "⚠️ GitHub bejelentkezés szükséges!" -ForegroundColor Red
    Write-Host "   Felhasználónév: GitHub username" -ForegroundColor White
    Write-Host "   Jelszó: Personal Access Token (Settings → Developer settings → Personal access tokens)" -ForegroundColor White
    Write-Host ""
    
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "===========================================" -ForegroundColor Green
        Write-Host "✅ SIKERES GITHUB FELTÖLTÉS!" -ForegroundColor Green
        Write-Host "===========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "🔗 Repository URL: https://github.com/toti85/project-s-agent-v2" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "📋 Következő lépések:" -ForegroundColor Yellow
        Write-Host "1. Ellenőrizd a repository-t GitHub-on" -ForegroundColor White
        Write-Host "2. Adj hozzá description-t: 'Advanced Autonomous AI Agent'" -ForegroundColor White
        Write-Host "3. Adj hozzá topics-okat: ai, autonomous-agent, python, browser-automation" -ForegroundColor White
        Write-Host "4. Hozz létre release-t: v2.0.0" -ForegroundColor White
        Write-Host "5. Oszd meg a közösséggel!" -ForegroundColor White
        Write-Host ""
        Write-Host "🎉 Project-S V2 sikeresen publikálva!" -ForegroundColor Green
    } else {
        throw "Git push sikertelen"
    }
    
} catch {
    Write-Host ""
    Write-Host "❌ Feltöltési hiba: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "🔧 Lehetséges megoldások:" -ForegroundColor Yellow
    Write-Host "1. Ellenőrizd a GitHub bejelentkezési adatokat" -ForegroundColor White
    Write-Host "2. Használj Personal Access Token-t jelszó helyett" -ForegroundColor White
    Write-Host "3. Ellenőrizd a hálózati kapcsolatot" -ForegroundColor White
    Write-Host "4. Próbáld meg GitHub Desktop-pal" -ForegroundColor White
    Write-Host ""
    Write-Host "💡 GitHub Desktop: https://desktop.github.com/" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Read-Host "Nyomj ENTER-t a kilépéshez"
