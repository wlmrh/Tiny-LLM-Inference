#!/bin/bash
set -e  # 出错即停止

echo "================================================="
echo "  🚀 开始自动化配置并编译 Tiny-LLM-Inference     "
echo "================================================="

# --- 智能判断是否需要 sudo ---
SUDO_CMD=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO_CMD="sudo"
    else
        echo "错误：您不是 root 用户，且系统未安装 sudo。请联系管理员。"
        exit 1
    fi
fi

# 1. 同步代码与子模块 (极其重要)
echo -e "\n[1/7] 更新 Git 子模块 (tokenizers-cpp 等)..."
git submodule update --init --recursive

# 2. 安装系统级依赖
echo -e "\n[2/7] 安装系统编译工具..."
$SUDO_CMD apt-get update
$SUDO_CMD apt-get install -y build-essential cmake git curl libssl-dev pkg-config python3-dev python3-venv

# 3. 安装 Rust 环境
echo -e "\n[3/7] 检查并安装 Rust..."
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # 在当前脚本中激活 Rust 环境变量
    source "$HOME/.cargo/env"
else
    echo "Rust 已存在。"
fi
# 确保脚本后续执行能找到 cargo
export CARGO_HOME="$HOME/.cargo"
export PATH="$CARGO_HOME/bin:$PATH"

# 4. 配置 Python 与 PyTorch (已配置国内加速源)
echo -e "\n[4/7] 配置 Python 虚拟环境与 PyTorch..."
python3 -m venv .venv
source .venv/bin/activate

# 提升 pip 下载速度 (清华源)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade pip

echo "正在从上海交大镜像源安装 PyTorch (CUDA 12.1)，这可能需要几分钟..."
pip install torch --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    pip install tokenizers transformers
fi

# 5. 提取并配置 CMake 与 CUDA 环境变量
echo -e "\n[5/7] 配置 CMake 与 CUDA 环境变量..."
TORCH_CMAKE_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')
export CMAKE_PREFIX_PATH=$TORCH_CMAKE_PATH
echo "✅ LibTorch 路径: $CMAKE_PREFIX_PATH"

# 解决 CMake 找不到 nvcc 的问题
if [ -f "/usr/local/cuda/bin/nvcc" ]; then
    export PATH=/usr/local/cuda/bin:$PATH
    export CUDACXX=/usr/local/cuda/bin/nvcc
    echo "✅ CUDA 编译器路径已设定: $CUDACXX"
else
    echo "⚠️ 警告: 未在 /usr/local/cuda/bin 找到 nvcc，如果后续编译报错请检查镜像！"
fi

# 6. 清理旧缓存
echo -e "\n[6/7] 清理旧的 CMake 缓存..."
rm -rf build

# 7. 编译项目
echo -e "\n[7/7] 开始编译 C++ 推理引擎..."
cmake -B build
# 使用所有 CPU 核心并行编译
cmake --build build -j$(nproc)

echo -e "\n================================================="
echo " 🎉 恭喜！环境配置与引擎编译已全部成功完成！"
echo " 后续再次登录服务器开发时，请务必先运行："
echo " source .venv/bin/activate && source ~/.cargo/env"
echo "================================================="