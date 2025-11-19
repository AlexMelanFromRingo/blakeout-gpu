# Blakeout GPU Mining для ALFIS

GPU-ускоренная версия алгоритма Blakeout для майнинга ALFIS на NVIDIA GPU.

## 🎯 Что это?

Это порт библиотеки [Blakeout](https://github.com/Revertron/blakeout) на CUDA для ускорения майнинга блоков в блокчейне ALFIS.

**Производительность:**
- **RTX 4080 SUPER:** ~1,682 H/s (3.7x быстрее CPU)
- **RTX 4090:** ~2,000-2,500 H/s  
- **RTX 3080:** ~800-1,000 H/s

## 📚 Документация

### Быстрый старт
- **Linux/MacOS:** [QUICK_START.md](QUICK_START.md)
- **Windows:** [WINDOWS_BUILD.md](WINDOWS_BUILD.md)

### Подробные руководства
- [ALFIS_GPU_INTEGRATION.md](ALFIS_GPU_INTEGRATION.md) - Интеграция с ALFIS
- [PERFORMANCE.md](blakeout-gpu/PERFORMANCE.md) - Анализ производительности
- [TEST_INSTRUCTIONS.md](blakeout-gpu/TEST_INSTRUCTIONS.md) - Тестирование

## 🚀 Быстрая установка

### Linux/MacOS

```bash
# Клонировать репозиторий
git clone https://github.com/YOUR_REPO/blakeout-gpu
cd blakeout-gpu

# Автоматическая сборка
chmod +x build_with_gpu.sh
./build_with_gpu.sh

# Запустить ALFIS
cd Alfis-master
./run_alfis_gpu.sh
```

### Windows

```powershell
# Клонировать репозиторий
git clone https://github.com/YOUR_REPO/blakeout-gpu
cd blakeout-gpu

# Автоматическая сборка
.\build_windows.ps1

# Запустить ALFIS
cd alfis-gpu-release
.\alfis.exe
```

## 📋 Требования

### Общие
- NVIDIA GPU с Compute Capability ≥ 6.0 (GTX 1000 series+)
- NVIDIA драйверы 450.00+
- CUDA Toolkit 11.0+ (для сборки)

### Linux
- GCC 7+
- CUDA Toolkit
- Rust 1.70+

### Windows
- Visual Studio Build Tools 2019+
- CUDA Toolkit  
- Rust (MSVC toolchain)

## 🏗️ Структура проекта

```
blakeout-gpu/
├── blakeout-gpu/          # CUDA библиотека Blakeout
│   ├── cuda/              # CUDA kernels (Blake2s, Blakeout)
│   ├── src/               # Rust FFI
│   └── build.rs           # CUDA compilation
├── Alfis-master/          # ALFIS с GPU поддержкой
│   └── src/gpu_miner.rs   # GPU mining интеграция
├── build_with_gpu.sh      # Linux/MacOS build script
├── build_windows.ps1      # Windows build script
└── docs/                  # Документация
```

## ⚙️ Технические детали

### Архитектура

**Blakeout** - memory-hard алгоритм хеширования:
- Основан на Blake2s (256-bit)
- 65,536 последовательных итераций
- 2MB буфер на хеш
- Спроектирован быть GPU-resistant

**GPU оптимизации:**
- Persistent GPU context (память выделяется один раз)
- Async memory operations (cudaMemcpyAsync)
- Optimal batch size: 4096 (8GB VRAM)
- Parallel processing across different nonces

### Почему ускорение только 3.7x?

Blakeout **специально спроектирован** быть GPU-resistant через:
- **65,536 последовательных** итераций Blake2s на каждый хеш
- Каждая итерация зависит от предыдущей (no parallelization)
- **2MB memory-hard** buffer на хеш

GPU может параллелить **разные nonces**, но не операции **внутри одного хеша**.

**3.7x - это отлично для memory-hard алгоритма!** Подробнее в [PERFORMANCE.md](blakeout-gpu/PERFORMANCE.md).

## 📊 Бенчмарки

### RTX 4080 SUPER

| Batch Size | Hash Rate | Time/Hash | VRAM Usage |
|------------|-----------|-----------|------------|
| 1024 | 443 H/s | 2.257ms | 2GB |
| 2048 | 885 H/s | 1.130ms | 4GB |
| **4096** | **1,682 H/s** | **0.595ms** | **8GB** ✅ |

**Сравнение с CPU:**
- Ryzen 5 5500 (12 потоков): 450 H/s
- GPU ускорение: **3.7x**

## 🙏 Благодарности

- [Revertron](https://github.com/Revertron) за ALFIS и Blakeout
- NVIDIA за CUDA Toolkit
- Rust и Cargo сообществу

---

**Made with ❤️ for ALFIS community**
