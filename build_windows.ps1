# ALFIS GPU Build Script for Windows
# Requires: Rust, CUDA Toolkit, Visual Studio Build Tools

param(
    [switch]$Clean = $false,
    [string]$CudaPath = "",
    [string]$OutputDir = "alfis-gpu-release"
)

$ErrorActionPreference = "Stop"

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "ALFIS GPU Build Script for Windows" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if command exists
function Test-Command {
    param($Command)
    try {
        if (Get-Command $Command -ErrorAction Stop) {
            return $true
        }
    } catch {
        return $false
    }
}

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

if (-not (Test-Command "cargo")) {
    Write-Host "ERROR: Rust (cargo) not found!" -ForegroundColor Red
    Write-Host "Please install Rust from https://rustup.rs/" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Rust installed" -ForegroundColor Green
cargo --version

if (-not (Test-Command "nvcc")) {
    Write-Host "ERROR: CUDA Toolkit (nvcc) not found!" -ForegroundColor Red
    Write-Host "Please install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads" -ForegroundColor Red
    exit 1
}
Write-Host "✓ CUDA Toolkit installed" -ForegroundColor Green
nvcc --version | Select-String "release"

if (Test-Command "nvidia-smi") {
    Write-Host "✓ NVIDIA Driver installed" -ForegroundColor Green
    $gpuInfo = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null | Select-Object -First 1
    if ($gpuInfo) {
        Write-Host "  GPU: $gpuInfo" -ForegroundColor Gray
    }
} else {
    Write-Host "⚠ nvidia-smi not found (driver may not be installed)" -ForegroundColor Yellow
}

# Find CUDA path
if ($CudaPath -eq "") {
    if ($env:CUDA_PATH) {
        $CudaPath = $env:CUDA_PATH
    } else {
        # Try to find CUDA in default location
        $cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if (Test-Path $cudaBase) {
            $versions = Get-ChildItem $cudaBase -Directory | Sort-Object Name -Descending
            if ($versions.Count -gt 0) {
                $CudaPath = $versions[0].FullName
            }
        }
    }
}

if ($CudaPath -and (Test-Path $CudaPath)) {
    Write-Host "✓ Using CUDA from: $CudaPath" -ForegroundColor Green
    $env:CUDA_PATH = $CudaPath
    $env:PATH += ";$CudaPath\bin"
} else {
    Write-Host "⚠ CUDA_PATH not set, using system default" -ForegroundColor Yellow
}

Write-Host ""

# Clean if requested
if ($Clean) {
    Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
    if (Test-Path "blakeout-gpu\target") {
        Remove-Item -Recurse -Force "blakeout-gpu\target"
    }
    if (Test-Path "Alfis-master\target") {
        Remove-Item -Recurse -Force "Alfis-master\target"
    }
    if (Test-Path $OutputDir) {
        Remove-Item -Recurse -Force $OutputDir
    }
    Write-Host "✓ Cleaned" -ForegroundColor Green
    Write-Host ""
}

# Build blakeout-gpu
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Step 1: Building blakeout-gpu library" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "blakeout-gpu"

Write-Host "Building blakeout-gpu with CUDA..." -ForegroundColor Yellow
cargo build --release
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to build blakeout-gpu" -ForegroundColor Red
    exit 1
}

# Find CUDA DLL
Write-Host "Searching for blakeout_cuda.dll..." -ForegroundColor Yellow
$cudaDll = Get-ChildItem -Recurse -Filter "blakeout_cuda.dll" "target\release\build\" 2>$null | Select-Object -First 1

if (-not $cudaDll) {
    Write-Host "ERROR: blakeout_cuda.dll not found!" -ForegroundColor Red
    Write-Host "CUDA compilation may have failed. Check build output above." -ForegroundColor Red
    exit 1
}

Write-Host "✓ Found CUDA library: $($cudaDll.FullName)" -ForegroundColor Green
$cudaDllDir = $cudaDll.Directory.FullName

Set-Location ".."

Write-Host ""

# Build ALFIS
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Step 2: Building ALFIS with GPU support" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "Alfis-master"

Write-Host "Building ALFIS..." -ForegroundColor Yellow
cargo build --release --features gpu --no-default-features
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to build ALFIS" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "target\release\alfis.exe")) {
    Write-Host "ERROR: alfis.exe not found!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ ALFIS built successfully" -ForegroundColor Green

Set-Location ".."

Write-Host ""

# Create release package
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Step 3: Creating release package" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path $OutputDir) {
    Remove-Item -Recurse -Force $OutputDir
}
New-Item -ItemType Directory -Path $OutputDir | Out-Null

Write-Host "Copying files to $OutputDir..." -ForegroundColor Yellow

# Copy ALFIS exe
Copy-Item "Alfis-master\target\release\alfis.exe" "$OutputDir\"
Write-Host "✓ Copied alfis.exe" -ForegroundColor Green

# Copy CUDA DLL
Copy-Item $cudaDll.FullName "$OutputDir\blakeout_cuda.dll"
Write-Host "✓ Copied blakeout_cuda.dll" -ForegroundColor Green

# Copy CUDA runtime DLL
if ($env:CUDA_PATH) {
    $cudartDll = Get-ChildItem "$env:CUDA_PATH\bin\cudart64_*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cudartDll) {
        Copy-Item $cudartDll.FullName "$OutputDir\"
        Write-Host "✓ Copied $($cudartDll.Name)" -ForegroundColor Green
    } else {
        Write-Host "⚠ cudart64_*.dll not found in CUDA_PATH\bin" -ForegroundColor Yellow
        Write-Host "  Users will need CUDA Toolkit or drivers installed" -ForegroundColor Yellow
    }
}

# Create README
$readmeContent = @"
ALFIS GPU Miner for Windows
============================

System Requirements:
- Windows 10/11 x64
- NVIDIA GPU (GTX 1000 series or newer)
- NVIDIA Drivers 450.00+

Usage:
------
1. Make sure your NVIDIA drivers are installed
2. Run: alfis.exe

The program will automatically detect and use your GPU for mining.

Performance:
------------
Expected hash rates (approximate):
- RTX 4090: ~2,500 H/s
- RTX 4080: ~1,700 H/s
- RTX 3090: ~1,200 H/s
- RTX 3080: ~1,000 H/s

Troubleshooting:
----------------
If you get "cudart64_XX.dll not found":
- Install NVIDIA GPU drivers from https://nvidia.com/drivers
- Or install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads

If GPU is not detected:
- Run: nvidia-smi
- Check that your GPU is visible
- Update drivers if needed

For more help, visit: https://github.com/YOUR_REPO

Built on: $(Get-Date -Format "yyyy-MM-dd HH:mm")
"@

Set-Content -Path "$OutputDir\README.txt" -Value $readmeContent
Write-Host "✓ Created README.txt" -ForegroundColor Green

# Create batch file for easy running
$batchContent = @"
@echo off
echo ALFIS GPU Miner
echo ===============
echo.
echo Checking GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ERROR: NVIDIA GPU or drivers not detected!
    echo Please install NVIDIA drivers from https://nvidia.com/drivers
    pause
    exit /b 1
)

echo Starting ALFIS with GPU support...
echo.
alfis.exe %*
"@

Set-Content -Path "$OutputDir\run_alfis.bat" -Value $batchContent
Write-Host "✓ Created run_alfis.bat" -ForegroundColor Green

Write-Host ""

# Show size
$totalSize = (Get-ChildItem $OutputDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "Package size: $([math]::Round($totalSize, 2)) MB" -ForegroundColor Cyan

Write-Host ""
Write-Host "====================================" -ForegroundColor Green
Write-Host "Build completed successfully! 🎉" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""
Write-Host "Release package created in: $OutputDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "Files included:" -ForegroundColor Yellow
Get-ChildItem $OutputDir | ForEach-Object {
    $size = if ($_.PSIsContainer) { "-" } else { "$([math]::Round($_.Length / 1KB, 1)) KB" }
    Write-Host "  $($_.Name.PadRight(25)) $size" -ForegroundColor Gray
}

Write-Host ""
Write-Host "To test locally:" -ForegroundColor Yellow
Write-Host "  cd $OutputDir" -ForegroundColor Gray
Write-Host "  .\alfis.exe" -ForegroundColor Gray
Write-Host ""
Write-Host "To create ZIP for distribution:" -ForegroundColor Yellow
Write-Host "  Compress-Archive -Path $OutputDir\* -DestinationPath alfis-gpu-windows-x64.zip" -ForegroundColor Gray
Write-Host ""
