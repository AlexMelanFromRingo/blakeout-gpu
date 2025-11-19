# Интеграция GPU Mining в ALFIS

## ✅ Что было сделано

1. **Оптимизирован batch_size**: изменен с 8192 на **4096** (оптимально для RTX 4080)
2. **Исправлена передача nonce**: теперь используется `current_nonce` вместо 0
3. **Исправлена логика результатов**: используются абсолютные nonce из GPU
4. **Добавлена оптимизация**: async memory operations в CUDA

## 🚀 Компиляция ALFIS с GPU

### Шаг 1: Убедитесь что CUDA установлен

```bash
nvcc --version
nvidia-smi
```

Должны работать оба. Если нет - установите [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

### Шаг 2: Скомпилируйте ALFIS с GPU feature

```bash
cd Alfis-master

# Release build с GPU поддержкой (без webgui)
cargo build --release --features gpu --no-default-features

# Или с webgui (требует системные библиотеки: libsoup-3.0, libwebkit2gtk-4.1, etc.)
# cargo build --release --features "webgui,doh,gpu"
```

**Примечание:** Если получаете ошибку про `libsoup-3.0`, используйте версию без webgui (первую команду).

### Шаг 3: Запустите ALFIS

```bash
./target/release/alfis
```

При запуске вы должны увидеть:

```
INFO GPU miner initialized successfully with batch size 4096
INFO Thread 0 using GPU for mining
```

## 📊 Ожидаемая производительность

### RTX 4080 SUPER (протестировано):
- **1,682 H/s** при batch_size=4096
- **3.7x быстрее** чем CPU (Ryzen 5 5500, 12 потоков, 450 H/s)
- **8GB VRAM** используется

### Другие карты (примерно):
- **RTX 4090**: ~2,000-2,500 H/s
- **RTX 4070**: ~1,000-1,200 H/s
- **RTX 3080**: ~800-1,000 H/s
- **RTX 3070**: ~600-800 H/s

## 🔧 Настройка

### Изменить batch_size

Если у вас меньше VRAM или хотите экспериментировать:

Отредактируйте `Alfis-master/src/gpu_miner.rs`:

```rust
impl Default for GpuMinerConfig {
    fn default() -> Self {
        GpuMinerConfig {
            batch_size: 2048, // Уменьшите для меньшего VRAM
            enabled: true,
        }
    }
}
```

**Memory usage по batch_size:**
- 1024 → 2GB VRAM
- 2048 → 4GB VRAM
- 4096 → 8GB VRAM (рекомендуется)
- 8192 → 16GB VRAM

### Отключить GPU mining

Либо скомпилируйте без feature:
```bash
cargo build --release
```

Либо в коде измените:
```rust
GpuMinerConfig {
    enabled: false,
    ...
}
```

## 🎯 Как это работает

1. **Thread 0** пытается майнить на GPU
2. Если GPU недоступен или отключен → автоматический **fallback на CPU**
3. Если GPU mining остановлен (ошибка) → **переключение на CPU**
4. **Другие threads** (1-N) всегда используют CPU

### Архитектура

```
ALFIS Miner
    ↓
Thread 0: GPU Mining (4096 hashes/batch)
    ├─ GPU available? → BlakeoutGpu::hash_batch()
    ├─ Success? → Return block
    └─ Fail/Stop? → Fallback to CPU

Threads 1-N: CPU Mining (traditional Blakeout)
```

## 🐛 Troubleshooting

### "No CUDA GPU available"

**Причины:**
- CUDA Toolkit не установлен
- `nvcc` не в PATH
- NVIDIA драйверы устарели
- GPU не поддерживается (нужна Compute Capability ≥ 6.0)

**Решение:**
```bash
# Проверить драйвер
nvidia-smi

# Проверить CUDA
nvcc --version

# Добавить в PATH (Linux)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### "CUDA error code: X"

**Причины:**
- Недостаточно VRAM
- Другое приложение использует GPU
- Thermal throttling

**Решение:**
- Уменьшите batch_size
- Закройте другие GPU приложения
- Проверьте температуру GPU

### GPU mining медленнее CPU?

**Причины:**
- batch_size слишком маленький
- GPU перегревается (throttling)
- Неправильная версия CUDA

**Решение:**
- Увеличьте batch_size до 4096
- Проверьте `nvidia-smi` на throttling
- Убедитесь что используете CUDA 11.0+

### Компиляция не находит nvcc

```bash
# Установите переменную окружения
export CUDA_PATH=/usr/local/cuda

# Или для конкретной версии
export CUDA_PATH=/usr/local/cuda-12.0
```

## 📈 Мониторинг

### Логи ALFIS

GPU mining выводит статистику каждые 10 секунд:

```
INFO GPU mining speed: 1682 H/s, max difficulty: 15, target: 20
```

### nvidia-smi

Проверьте загрузку GPU:

```bash
watch -n 1 nvidia-smi
```

Ищите:
- **GPU Utilization**: должно быть ~95-100%
- **Memory Usage**: ~8GB для batch_size=4096
- **Temperature**: <80°C оптимально
- **Power**: близко к TDP

## 🔬 Бенчмаркинг

Для тестирования производительности используйте standalone примеры:

```bash
cd blakeout-gpu

# Тест производительности на разных batch sizes
cargo run --release --example perf_test

# Тест реального майнинга
cargo run --release --example gpu_miner
```

## 📚 Технические детали

### Почему ускорение только 3.7x?

Blakeout спроектирован быть **GPU-resistant**:
- 65,536 **последовательных** итераций Blake2s на хеш
- Каждая итерация ждет предыдущую (no parallelization)
- 2MB buffer на хеш (memory-hard)

GPU может параллелить **разные nonce**, но не **внутри одного хеша**.

Подробнее в `blakeout-gpu/PERFORMANCE.md`.

### Оптимизации в CUDA коде

1. **Persistent GPU Context** - память выделяется один раз при старте
2. **Async Memory Operations** - cudaMemcpyAsync для overlap
3. **Optimal Thread Configuration** - 256 threads/block
4. **Pre-allocated Buffers** - 8GB буфер для всех nonces

## 🎓 Для разработчиков

### Структура кода

```
blakeout-gpu/
├── cuda/
│   ├── blake2s.cu        # Blake2s implementation
│   ├── blakeout.cu       # Blakeout kernel + context API
│   └── blake2s.cuh       # Headers
├── src/
│   ├── lib.rs            # BlakeoutGpu struct
│   └── gpu.rs            # FFI bindings to CUDA
└── build.rs              # CUDA compilation

Alfis-master/src/
├── gpu_miner.rs          # GPU mining logic
└── miner.rs              # Main miner (CPU + GPU)
```

### Изменения в ALFIS

**gpu_miner.rs:**
- Используйте `hasher.hash_batch(data, current_nonce, difficulty)`
- `current_nonce` передается в GPU для правильных результатов
- Результаты содержат абсолютные nonce значения

**miner.rs:**
- Thread 0 пытается GPU первым
- Fallback на CPU при ошибках
- Lazy static для единственного GPU instance

## 📝 Changelog

### Latest (2025-11-19)

- ✅ Оптимизирован batch_size: 8192 → 4096
- ✅ Исправлен bug с nonce (использовался 0 вместо current_nonce)
- ✅ Добавлены async memory operations
- ✅ Улучшена обработка ошибок в CUDA
- ✅ Добавлена документация производительности

### Previous

- ✅ Persistent GPU context (избегает malloc overhead)
- ✅ Интеграция с ALFIS miner
- ✅ Автоматический CPU fallback

## 🤝 Вклад

Если вы хотите улучшить GPU mining:

1. Экспериментируйте с batch_size
2. Тестируйте на разных GPU
3. Профилируйте с NVIDIA Nsight
4. Предлагайте оптимизации CUDA кода

## 📞 Поддержка

- Issues: https://github.com/Revertron/Alfis/issues
- ALFIS Community: https://alfis.name
- CUDA Documentation: https://docs.nvidia.com/cuda/
