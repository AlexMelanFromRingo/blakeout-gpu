# Сборка ALFIS с GPU на Windows

## 📋 Требования

### 1. Установите инструменты разработки

**Visual Studio Build Tools:**
- Скачайте [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
- При установке выберите "Desktop development with C++"

**Rust:**
- Скачайте [rustup](https://rustup.rs/)
- Установите: `rustup-init.exe`
- Выберите установку `x86_64-pc-windows-msvc`

**CUDA Toolkit:**
- Скачайте [CUDA Toolkit 11.8+](https://developer.nvidia.com/cuda-downloads)
- Установите с default настройками
- CUDA будет в `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`

**Git:**
- Скачайте [Git for Windows](https://git-scm.com/download/win)

### 2. Проверьте установку

Откройте PowerShell или CMD:

```powershell
# Проверка Rust
cargo --version
rustc --version

# Проверка CUDA
nvcc --version

# Проверка GPU
nvidia-smi
```

## 🔨 Автоматическая сборка

Я создал скрипт для автоматической сборки на Windows:

### PowerShell скрипт

```powershell
# Скачайте репозиторий
git clone https://github.com/YOUR_REPO/blakeout-gpu
cd blakeout-gpu

# Запустите сборку
.\build_windows.ps1
```

Скрипт автоматически:
1. ✅ Проверит наличие всех инструментов
2. ✅ Соберет blakeout-gpu с CUDA
3. ✅ Соберет ALFIS с GPU
4. ✅ Скопирует все необходимые DLL
5. ✅ Создаст готовую папку для распространения

## 🔧 Ручная сборка

### Шаг 1: Установите переменные окружения

В PowerShell:

```powershell
# Найдите CUDA путь (обычно C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x)
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
$env:PATH += ";$env:CUDA_PATH\bin"

# Для Visual Studio 2022 (найдите вашу версию)
$env:PATH += ";C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.xx.xxxxx\bin\Hostx64\x64"
```

### Шаг 2: Соберите blakeout-gpu

```powershell
cd blakeout-gpu

# Clean build
cargo clean

# Release сборка с CUDA
cargo build --release

# Проверьте что CUDA библиотека создана
Get-ChildItem -Recurse -Filter "blakeout_cuda.dll" target\release\build\
```

### Шаг 3: Соберите ALFIS

```powershell
cd ..\Alfis-master

# Clean build
cargo clean

# Release сборка с GPU (без webgui для простоты)
cargo build --release --features gpu --no-default-features
```

### Шаг 4: Соберите все файлы

```powershell
# Создайте папку для распространения
mkdir alfis-gpu-release

# Скопируйте exe
copy target\release\alfis.exe alfis-gpu-release\

# Найдите и скопируйте CUDA DLL
$cudaDll = Get-ChildItem -Recurse -Filter "blakeout_cuda.dll" ..\blakeout-gpu\target\release\build\ | Select-Object -First 1
copy $cudaDll.FullName alfis-gpu-release\

# Скопируйте CUDA runtime (если нужно)
copy "$env:CUDA_PATH\bin\cudart64_*.dll" alfis-gpu-release\

# Скопируйте конфигурацию (если есть)
# copy alfis.toml alfis-gpu-release\
```

## 📦 Создание распространяемого пакета

### Вариант 1: ZIP архив

```powershell
# Создайте ZIP
Compress-Archive -Path alfis-gpu-release\* -DestinationPath alfis-gpu-windows-x64.zip
```

### Вариант 2: Installer с WiX

Установите WiX Toolset:
```powershell
cargo install cargo-wix
```

Создайте installer:
```powershell
cd Alfis-master
cargo wix --features gpu --no-default-features
```

## 🔍 Поиск зависимостей

Чтобы найти все необходимые DLL:

```powershell
# Установите Dependencies Walker или используйте встроенный dumpbin
dumpbin /dependents target\release\alfis.exe

# Или используйте PowerShell
Get-Command target\release\alfis.exe | Select-Object -ExpandProperty FileVersionInfo | Select-Object -ExpandProperty OriginalFilename
```

Обычно нужны:
- `blakeout_cuda.dll` (из сборки blakeout-gpu)
- `cudart64_XX.dll` (CUDA runtime)
- Возможно `cublas64_XX.dll`, `cublasLt64_XX.dll` (если используются)

## ⚙️ Статическая линковка (сложно)

Для полностью автономного exe без DLL:

### Вариант 1: Статический CUDA runtime

В `blakeout-gpu/build.rs` добавьте:

```rust
println!("cargo:rustc-link-arg=/NODEFAULTLIB:cudart.lib");
println!("cargo:rustc-link-lib=static=cudart_static");
```

### Вариант 2: Встроить DLL в exe

Используйте [include-flate](https://crates.io/crates/include-flate) для встраивания:

1. Добавьте в `Cargo.toml`:
```toml
[dependencies]
include-flate = "0.2"
```

2. Встройте DLL в бинарник и извлекайте при запуске

**Примечание:** Статическая линковка CUDA очень сложна и часто не работает из-за лицензионных ограничений NVIDIA.

## 🚀 Запуск

### С DLL в той же папке:

```powershell
cd alfis-gpu-release
.\alfis.exe
```

### С DLL в другой папке:

```powershell
$env:PATH += ";C:\path\to\cuda\dlls"
.\alfis.exe
```

## 🐛 Troubleshooting

### "blakeout_cuda.dll not found"

**Решение 1:** Скопируйте DLL в папку с exe
```powershell
$cudaDll = Get-ChildItem -Recurse -Filter "blakeout_cuda.dll" ..\blakeout-gpu\target\release\build\ | Select-Object -First 1
copy $cudaDll.FullName .\
```

**Решение 2:** Добавьте в PATH
```powershell
$cudaDllPath = (Get-ChildItem -Recurse -Filter "blakeout_cuda.dll" ..\blakeout-gpu\target\release\build\ | Select-Object -First 1).Directory.FullName
$env:PATH += ";$cudaDllPath"
```

### "cudart64_XX.dll not found"

```powershell
copy "$env:CUDA_PATH\bin\cudart64_*.dll" .\
```

### "CUDA error during initialization"

1. Проверьте драйверы:
```powershell
nvidia-smi
```

2. Проверьте compute capability вашей GPU:
```powershell
nvidia-smi --query-gpu=compute_cap --format=csv
```

3. Если нужно изменить target architecture, установите переменную окружения:
```powershell
$env:CUDA_COMPUTE_ARCH = "sm_89"  # Для RTX 4090
$env:CUDA_COMPUTE_ARCH = "sm_86"  # Для RTX 3080/4080
```

## 📊 Размер итогового пакета

Примерный размер:
- `alfis.exe`: ~2-5 MB
- `blakeout_cuda.dll`: ~1-2 MB
- `cudart64_XX.dll`: ~0.5 MB
- **Итого:** ~4-8 MB

С installer WiX: ~5-10 MB

## 🎯 Рекомендации

### Для разработки:
- Используйте ручную сборку
- Держите DLL в build директории
- Используйте PowerShell скрипты для удобства

### Для распространения:
- Создайте ZIP с exe + все DLL
- Или создайте installer с WiX
- Включите README с системными требованиями

### Для минимального размера:
- Используйте `strip = true` в Cargo.toml (уже включено)
- Используйте UPX для сжатия exe (опционально)
- Статически линкуйте что можно

## 📝 Системные требования для конечных пользователей

Ваш пакет будет требовать:
- ✅ Windows 10/11 x64
- ✅ NVIDIA GPU с Compute Capability ≥ 6.0 (GTX 1000 series+)
- ✅ NVIDIA драйверы 450.00+
- ✅ Visual C++ Redistributable 2015-2022 (обычно уже установлен)
- ❌ **НЕ требуется** CUDA Toolkit (если включить runtime DLL)

## 🔄 Обновление

При обновлении кода:

```powershell
git pull
.\build_windows.ps1  # Пересоберет все
```

## 📚 Дополнительные ресурсы

- [CUDA Windows Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [Rust Windows MSVC](https://rust-lang.github.io/rustup/installation/windows.html)
- [Cargo WiX](https://github.com/volks73/cargo-wix)
