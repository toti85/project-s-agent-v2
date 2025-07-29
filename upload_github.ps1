# GitHub Upload PowerShell Script - Project-S V2
# ===============================================

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "GitHub Upload Script - Project-S V2" -ForegroundColor Yellow
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Git check
$gitInstalled = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitInstalled) {
    Write-Host "Error: Git is not installed!" -ForegroundColor Red
    Write-Host "Please run install_git.ps1 first" -ForegroundColor Yellow
    Read-Host "Press ENTER to exit"
    exit 1
}

Write-Host "Git found:" -ForegroundColor Green
git --version
Write-Host ""

# Check current directory
$currentPath = Get-Location
Write-Host "Current directory: $currentPath" -ForegroundColor Blue

# Check Project-S V2 files
$requiredFiles = @("README.md", "main.py", "complex_task_tester.py", "requirements.txt", "LICENSE")
$missingFiles = @()

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "Missing files: $($missingFiles -join ', ')" -ForegroundColor Red
    Write-Host "Please navigate to c:\v2_architecture\project-s-clean" -ForegroundColor Yellow
    Read-Host "Press ENTER to exit"
    exit 1
}

Write-Host "All required files present" -ForegroundColor Green
Write-Host ""

# Git configuration check
Write-Host "Checking Git configuration..." -ForegroundColor Blue

$gitUser = git config --global user.name
$gitEmail = git config --global user.email

if (-not $gitUser -or -not $gitEmail) {
    Write-Host "Git configuration missing" -ForegroundColor Yellow
    Write-Host ""
    
    if (-not $gitUser) {
        $userName = Read-Host "Enter your GitHub username"
        git config --global user.name "$userName"
        Write-Host "Username set: $userName" -ForegroundColor Green
    }
    
    if (-not $gitEmail) {
        $userEmail = Read-Host "Enter your GitHub email"
        git config --global user.email "$userEmail"
        Write-Host "Email set: $userEmail" -ForegroundColor Green
    }
    Write-Host ""
} else {
    Write-Host "Git configuration OK:" -ForegroundColor Green
    Write-Host "   Name: $gitUser" -ForegroundColor White
    Write-Host "   Email: $gitEmail" -ForegroundColor White
    Write-Host ""
}

# GitHub upload confirmation
Write-Host "Ready for GitHub upload!" -ForegroundColor Green
Write-Host ""
Write-Host "Repository: https://github.com/toti85/project-s-agent-v2" -ForegroundColor Cyan
Write-Host ""

$confirm = Read-Host "Upload now? (y/n)"
if ($confirm -ne 'y' -and $confirm -ne 'yes') {
    Write-Host "Upload cancelled" -ForegroundColor Yellow
    Read-Host "Press ENTER to exit"
    exit 0
}

Write-Host ""
Write-Host "GitHub upload in progress..." -ForegroundColor Blue
Write-Host ""

try {
    # Initialize Git repository
    Write-Host "1. Initializing Git repository..." -ForegroundColor Yellow
    git init
    if ($LASTEXITCODE -ne 0) { throw "Git init failed" }
    Write-Host "Git init completed" -ForegroundColor Green
    
    # Add remote repository
    Write-Host "2. Adding remote repository..." -ForegroundColor Yellow
    git remote add origin https://github.com/toti85/project-s-agent-v2.git 2>$null
    Write-Host "Remote origin set" -ForegroundColor Green
    
    # Add files
    Write-Host "3. Adding files..." -ForegroundColor Yellow
    git add .
    if ($LASTEXITCODE -ne 0) { throw "Git add failed" }
    Write-Host "All files added" -ForegroundColor Green
    
    # Create commit
    Write-Host "4. Creating commit..." -ForegroundColor Yellow
    git commit -m "Initial commit: Project-S V2 - Advanced Autonomous AI Agent (100% tested)"
    if ($LASTEXITCODE -ne 0) { throw "Git commit failed" }
    Write-Host "Commit created" -ForegroundColor Green
    
    # Set main branch
    Write-Host "5. Setting main branch..." -ForegroundColor Yellow
    git branch -M main
    Write-Host "Main branch set" -ForegroundColor Green
    
    # Push to GitHub
    Write-Host "6. Pushing to GitHub..." -ForegroundColor Yellow
    Write-Host "GitHub login required!" -ForegroundColor Red
    Write-Host "   Username: GitHub username" -ForegroundColor White
    Write-Host "   Password: Personal Access Token" -ForegroundColor White
    Write-Host ""
    
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "===========================================" -ForegroundColor Green
        Write-Host "SUCCESSFUL GITHUB UPLOAD!" -ForegroundColor Green
        Write-Host "===========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Repository URL: https://github.com/toti85/project-s-agent-v2" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "1. Check repository on GitHub" -ForegroundColor White
        Write-Host "2. Add description: 'Advanced Autonomous AI Agent'" -ForegroundColor White
        Write-Host "3. Add topics: ai, autonomous-agent, python, browser-automation" -ForegroundColor White
        Write-Host "4. Create release: v2.0.0" -ForegroundColor White
        Write-Host "5. Share with community!" -ForegroundColor White
        Write-Host ""
        Write-Host "Project-S V2 successfully published!" -ForegroundColor Green
    } else {
        throw "Git push failed"
    }
    
} catch {
    Write-Host ""
    Write-Host "Upload error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Possible solutions:" -ForegroundColor Yellow
    Write-Host "1. Check GitHub credentials" -ForegroundColor White
    Write-Host "2. Use Personal Access Token as password" -ForegroundColor White
    Write-Host "3. Check network connection" -ForegroundColor White
    Write-Host "4. Try GitHub Desktop" -ForegroundColor White
    Write-Host ""
    Write-Host "GitHub Desktop: https://desktop.github.com/" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Read-Host "Press ENTER to exit"
