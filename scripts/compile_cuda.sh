#!/bin/bash
# Manual CUDA kernel compilation script

set -e

echo "🔧 Compiling CUDA kernels for Blakeout GPU"
echo ""

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ Error: nvcc not found in PATH"
    echo "Please install CUDA Toolkit and ensure nvcc is available"
    echo ""
    echo "Installation:"
    echo "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"
    echo "  Or download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Show nvcc version
echo "📌 Using nvcc:"
nvcc --version | head -n 4
echo ""

# Detect GPU architecture
echo "🔍 Detecting GPU architecture..."
if command -v nvidia-smi &> /dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    if [ -n "$COMPUTE_CAP" ]; then
        ARCH="sm_$COMPUTE_CAP"
        echo "✅ Detected: $ARCH"
    else
        ARCH="sm_86"
        echo "⚠️  Could not detect GPU, using default: $ARCH"
    fi
else
    ARCH="sm_86"
    echo "⚠️  nvidia-smi not found, using default: $ARCH"
fi
echo ""

# Create src directory if it doesn't exist
mkdir -p src

# Compile main PTX
echo "🚀 Compiling main kernel (src/blake2s.ptx)..."
nvcc -ptx src/blake2s.cu -o src/blake2s.ptx \
    --gpu-architecture=$ARCH \
    -O3 \
    --use_fast_math \
    --maxrregcount=64 \
    -Xptxas -v \
    --default-stream per-thread

if [ $? -eq 0 ]; then
    echo "✅ Main kernel compiled successfully"
else
    echo "❌ Failed to compile main kernel"
    exit 1
fi
echo ""

# Compile for multiple architectures
echo "📦 Compiling for multiple architectures..."
ARCHITECTURES=("sm_70" "sm_75" "sm_80" "sm_86" "sm_89")

for arch in "${ARCHITECTURES[@]}"; do
    output="src/blake2s_${arch#sm_}.ptx"
    echo "  Compiling for $arch..."
    
    nvcc -ptx src/blake2s.cu -o "$output" \
        --gpu-architecture=$arch \
        -O3 \
        --use_fast_math \
        2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "  ✅ $arch compiled"
    else
        echo "  ⚠️  $arch compilation skipped (not supported)"
    fi
done

echo ""
echo "🎉 CUDA kernel compilation complete!"
echo ""
echo "Generated files:"
ls -lh src/*.ptx 2>/dev/null || echo "No PTX files found"
echo ""
echo "You can now run: cargo build --release"