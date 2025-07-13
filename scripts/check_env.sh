#!/bin/bash
# 环境检查脚本

echo "=== 环境检查 ==="
echo

echo "1. Python 环境:"
echo "   Python: $(which python)"
echo "   Version: $(python --version 2>&1)"
echo "   Virtual env: ${VIRTUAL_ENV:-Not activated}"
echo

echo "2. Pytest:"
echo "   Pytest: $(which pytest)"
if [ -f ".venv/bin/pytest" ]; then
    echo "   虚拟环境 pytest: 存在 ✓"
else
    echo "   虚拟环境 pytest: 不存在 ✗"
fi
echo

echo "3. 包安装状态:"
if pip list | grep -q "knowledge-core-engine"; then
    echo "   knowledge-core-engine: 已安装 ✓"
else
    echo "   knowledge-core-engine: 未安装 ✗"
    echo "   请运行: pip install -e ."
fi
echo

echo "4. .env 文件检查:"
if [ -f ".env" ]; then
    echo "   .env 文件: 存在 ✓"
    
    # 检查是否有内联注释
    if grep -E "=.*#" .env > /dev/null 2>&1; then
        echo "   警告: .env 文件中发现内联注释！"
        echo "   问题行:"
        grep -E "=.*#" .env | head -5
        echo "   请删除值后面的注释"
    else
        echo "   内联注释: 无 ✓"
    fi
else
    echo "   .env 文件: 不存在 ✗"
    echo "   请运行: cp .env.example .env"
fi
echo

echo "5. 测试运行建议:"
echo "   使用: .venv/bin/pytest tests/unit -v"
echo "   或者: python -m pytest tests/unit -v"