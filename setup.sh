#!/bin/bash
set -e  # 出错即停止

echo "开始配置 Tiny-LLM-Inference 环境..."

# 1. 安装系统级依赖 (针对 Ubuntu/Debian 服务器)
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl libssl-dev pkg-config python3-dev python3-venv

# 2. 安装 Rust 环境
if ! command -v cargo &> /dev/null; then
    echo "安装 Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "Rust 已存在，跳过安装。"
fi

# 3. 创建并配置 Python 虚拟环境
echo "配置 Python 虚拟环境..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    pip install tokenizers transformers
fi

# 4. 设置环境变量 (解决你之前遇到的路径问题)
export CARGO_HOME="$HOME/.cargo"
export PATH="$CARGO_HOME/bin:$PATH"
# 自动探测 site-packages 路径并写入 .env 或脚本
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
export PYTHONPATH=$PYTHONPATH:$(pwd)/.venv/lib/python$PYTHON_VERSION/site-packages

# 5. 更新项目子模块 (解决 tokenizers-cpp 源码问题)
echo "同步子模块..."
git submodule update --init --recursive

echo "环境配置完成！"
echo "请运行 'source .venv/bin/activate' 和 'source \$HOME/.cargo/env' 开始开发。"