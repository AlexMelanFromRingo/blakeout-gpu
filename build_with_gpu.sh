#!/bin/bash
# Script to build ALFIS with GPU support
# Make sure CUDA is installed and nvcc is in PATH

set -e  # Exit on error

echo "==================================="
echo "ALFIS GPU Build Script"
echo "==================================="
echo ""

# Check for CUDA
echo "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found in PATH"
    echo ""
    echo "Please install CUDA Toolkit and add nvcc to PATH:"
    echo "  export PATH=/usr/local/cuda/bin:\$PATH"
    echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    echo ""
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "✓ Found nvcc version: $NVCC_VERSION"

# Check for nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found"
else
    echo "✓ Found nvidia-smi"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
fi

echo ""
echo "==================================="
echo "Step 1: Build blakeout-gpu library"
echo "==================================="
echo ""

cd blakeout-gpu

# Clean previous build
echo "Cleaning previous build..."
cargo clean

# Build blakeout-gpu with CUDA
echo "Building blakeout-gpu with CUDA..."
cargo build --release

# Check if CUDA library was created
CUDA_LIB=$(find target/release/build -name "libblakeout_cuda.so" 2>/dev/null | head -1)
if [ -z "$CUDA_LIB" ]; then
    echo ""
    echo "ERROR: libblakeout_cuda.so was not created!"
    echo "CUDA compilation may have failed."
    echo ""
    exit 1
fi

echo "✓ CUDA library created: $CUDA_LIB"
CUDA_LIB_DIR=$(dirname "$CUDA_LIB")
echo "✓ Library directory: $CUDA_LIB_DIR"

echo ""
echo "==================================="
echo "Step 2: Build ALFIS with GPU support"
echo "==================================="
echo ""

cd ../Alfis-master

# Clean previous build
echo "Cleaning previous build..."
cargo clean

# Build ALFIS with GPU feature
echo "Building ALFIS with GPU feature..."
cargo build --release --features gpu --no-default-features

# Check if alfis binary was created
if [ ! -f "target/release/alfis" ]; then
    echo ""
    echo "ERROR: ALFIS binary was not created!"
    echo ""
    exit 1
fi

echo "✓ ALFIS binary created: target/release/alfis"

echo ""
echo "==================================="
echo "Step 3: Create run script"
echo "==================================="
echo ""

# Create a run script that sets LD_LIBRARY_PATH
cat > run_alfis_gpu.sh << 'EOF'
#!/bin/bash
# Run ALFIS with GPU support

# Find the CUDA library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_LIB=$(find "$SCRIPT_DIR/../blakeout-gpu/target/release/build" -name "libblakeout_cuda.so" 2>/dev/null | head -1)

if [ -z "$CUDA_LIB" ]; then
    echo "ERROR: libblakeout_cuda.so not found!"
    echo "Please run build_with_gpu.sh first"
    exit 1
fi

CUDA_LIB_DIR=$(dirname "$CUDA_LIB")

# Set library path and run ALFIS
export LD_LIBRARY_PATH="$CUDA_LIB_DIR:$LD_LIBRARY_PATH"
echo "Using CUDA library from: $CUDA_LIB_DIR"
echo ""

exec "$SCRIPT_DIR/target/release/alfis" "$@"
EOF

chmod +x run_alfis_gpu.sh

echo "✓ Created run script: Alfis-master/run_alfis_gpu.sh"

echo ""
echo "==================================="
echo "Build completed successfully! 🎉"
echo "==================================="
echo ""
echo "To run ALFIS with GPU support:"
echo ""
echo "  cd Alfis-master"
echo "  ./run_alfis_gpu.sh"
echo ""
echo "Or set LD_LIBRARY_PATH manually:"
echo ""
echo "  export LD_LIBRARY_PATH=$CUDA_LIB_DIR:\$LD_LIBRARY_PATH"
echo "  ./target/release/alfis"
echo ""
