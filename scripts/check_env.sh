#!/bin/bash
# Environment check script for Blakeout GPU

echo "🔍 Checking Blakeout GPU build environment..."
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check counter
CHECKS_PASSED=0
CHECKS_FAILED=0

check_command() {
    local cmd=$1
    local name=$2
    local required=$3
    
    if command -v $cmd &> /dev/null; then
        echo -e "${GREEN}✓${NC} $name is installed"
        VERSION=$($cmd --version 2>&1 | head -n 1)
        echo "  Version: $VERSION"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
        return 0
    else
        if [ "$required" = "required" ]; then
            echo -e "${RED}✗${NC} $name is NOT installed (REQUIRED)"
            CHECKS_FAILED=$((CHECKS_FAILED + 1))
        else
            echo -e "${YELLOW}!${NC} $name is NOT installed (optional)"
        fi
        return 1
    fi
}

echo "=== Essential Tools ==="
check_command rustc "Rust compiler" required
check_command cargo "Cargo" required
check_command nvcc "NVIDIA CUDA Compiler" required
echo ""

echo "=== GPU Information ==="
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} nvidia-smi is available"
    echo ""
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
    echo ""
    
    # Check compute capability
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')
    if [ -n "$COMPUTE_CAP" ]; then
        COMPUTE_NUM=$(echo $COMPUTE_CAP | tr -d '.')
        if [ "$COMPUTE_NUM" -ge 70 ]; then
            echo -e "${GREEN}✓${NC} Compute capability $COMPUTE_CAP is supported (>= 7.0)"
        else
            echo -e "${YELLOW}!${NC} Compute capability $COMPUTE_CAP may have limited support (< 7.0)"
        fi
    fi
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo -e "${RED}✗${NC} nvidia-smi is NOT available"
    echo "  Cannot detect GPU information"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi
echo ""

echo "=== CUDA Environment ==="
if [ -n "$CUDA_PATH" ]; then
    echo -e "${GREEN}✓${NC} CUDA_PATH is set: $CUDA_PATH"
elif [ -n "$CUDA_HOME" ]; then
    echo -e "${GREEN}✓${NC} CUDA_HOME is set: $CUDA_HOME"
elif [ -d "/usr/local/cuda" ]; then
    echo -e "${YELLOW}!${NC} CUDA_PATH not set, but /usr/local/cuda exists"
    echo "  Consider: export CUDA_PATH=/usr/local/cuda"
else
    echo -e "${RED}✗${NC} CUDA_PATH is not set and /usr/local/cuda not found"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi

# Check CUDA libraries
if [ -f "/usr/local/cuda/lib64/libcudart.so" ]; then
    echo -e "${GREEN}✓${NC} CUDA runtime library found"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo -e "${YELLOW}!${NC} CUDA runtime library not found at default location"
fi
echo ""

echo "=== PATH Configuration ==="
if echo $PATH | grep -q "cuda"; then
    echo -e "${GREEN}✓${NC} CUDA binaries are in PATH"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo -e "${YELLOW}!${NC} CUDA binaries may not be in PATH"
    echo "  Consider adding: export PATH=/usr/local/cuda/bin:\$PATH"
fi

if echo $LD_LIBRARY_PATH | grep -q "cuda"; then
    echo -e "${GREEN}✓${NC} CUDA libraries are in LD_LIBRARY_PATH"
else
    echo -e "${YELLOW}!${NC} CUDA libraries may not be in LD_LIBRARY_PATH"
    echo "  Consider adding: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
fi
echo ""

echo "=== Project Files ==="
if [ -f "src/blake2s.cu" ]; then
    echo -e "${GREEN}✓${NC} CUDA kernel source found (src/blake2s.cu)"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo -e "${RED}✗${NC} CUDA kernel source NOT found (src/blake2s.cu)"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi

if [ -f "Cargo.toml" ]; then
    echo -e "${GREEN}✓${NC} Cargo.toml found"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo -e "${RED}✗${NC} Cargo.toml NOT found"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi

if [ -f "src/blake2s.ptx" ]; then
    echo -e "${GREEN}✓${NC} Compiled PTX found (src/blake2s.ptx)"
    PTX_SIZE=$(ls -lh src/blake2s.ptx | awk '{print $5}')
    echo "  Size: $PTX_SIZE"
else
    echo -e "${YELLOW}!${NC} Compiled PTX NOT found (needs compilation)"
    echo "  Run: ./scripts/compile_cuda.sh"
fi
echo ""

echo "=== Summary ==="
echo "Checks passed: ${GREEN}$CHECKS_PASSED${NC}"
echo "Checks failed: ${RED}$CHECKS_FAILED${NC}"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 Environment is ready!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Compile CUDA kernel: ./scripts/compile_cuda.sh"
    echo "  2. Build project: cargo build --release"
    echo "  3. Run tests: cargo test --release"
    exit 0
else
    echo -e "${RED}⚠️  Some checks failed. Please fix the issues above.${NC}"
    echo ""
    echo "Common fixes:"
    echo "  - Install CUDA: https://developer.nvidia.com/cuda-downloads"
    echo "  - Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "  - Set environment variables:"
    echo "      export CUDA_PATH=/usr/local/cuda"
    echo "      export PATH=\$CUDA_PATH/bin:\$PATH"
    echo "      export LD_LIBRARY_PATH=\$CUDA_PATH/lib64:\$LD_LIBRARY_PATH"
    exit 1
fi