#!/usr/bin/env python
"""
冒烟测试脚本 - 快速验证系统基本功能

使用方法：
    python tests/smoke_test.py
    
返回码：
    0 - 所有测试通过
    1 - 有测试失败
"""

import asyncio
import sys
import os
from pathlib import Path
import tempfile
import shutil
from typing import List, Tuple

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_core_engine import KnowledgeEngine


class Colors:
    """终端颜色"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(title: str):
    """打印标题"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*50}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title:^50}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*50}{Colors.ENDC}\n")


def print_test(name: str, passed: bool, message: str = ""):
    """打印测试结果"""
    status = f"{Colors.GREEN}✓ PASS{Colors.ENDC}" if passed else f"{Colors.RED}✗ FAIL{Colors.ENDC}"
    print(f"  {name:<40} {status}")
    if message and not passed:
        print(f"    {Colors.YELLOW}→ {message}{Colors.ENDC}")


async def test_basic_flow() -> Tuple[bool, str]:
    """测试基本流程"""
    try:
        # 创建临时目录
        test_dir = Path(tempfile.mkdtemp()) / "smoke_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试文档
        test_file = test_dir / "test.md"
        test_file.write_text("""
# 测试文档

这是一个测试文档，用于验证RAG系统的基本功能。

## 什么是冒烟测试？
冒烟测试是一种快速验证系统基本功能的测试方法。
""")
        
        # 创建引擎
        engine = KnowledgeEngine(
            persist_directory=str(test_dir / "vector_db")
        )
        
        # 添加文档
        result = await engine.add(str(test_file))
        if result["processed_files"] != 1:
            return False, f"文档处理失败: {result}"
        
        # 查询
        answer = await engine.ask("什么是冒烟测试？")
        if not answer or "冒烟测试" not in answer:
            return False, f"查询失败或答案不相关: {answer[:100]}"
        
        # 清理
        await engine.close()
        shutil.rmtree(test_dir.parent)
        
        return True, ""
        
    except Exception as e:
        return False, str(e)


async def test_multiple_files() -> Tuple[bool, str]:
    """测试多文件处理"""
    try:
        test_dir = Path(tempfile.mkdtemp()) / "multi_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建多个文件
        files = []
        for i in range(3):
            file = test_dir / f"doc{i}.txt"
            file.write_text(f"这是第{i+1}个测试文档。")
            files.append(str(file))
        
        # 处理
        engine = KnowledgeEngine()
        result = await engine.add(files)
        
        if result["processed_files"] != 3:
            return False, f"期望处理3个文件，实际: {result['processed_files']}"
        
        await engine.close()
        shutil.rmtree(test_dir.parent)
        
        return True, ""
        
    except Exception as e:
        return False, str(e)


async def test_error_handling() -> Tuple[bool, str]:
    """测试错误处理"""
    try:
        engine = KnowledgeEngine()
        
        # 测试不存在的文件
        result = await engine.add("non_existent_file.txt")
        if result["processed_files"] != 0 or len(result["failed_files"]) != 1:
            return False, "应该正确处理不存在的文件"
        
        # 测试空查询 - 应该优雅地处理
        try:
            answer = await engine.ask("")
            # 某些系统可能返回错误消息而不是抛出异常
        except ValueError:
            # 预期的行为 - 空查询应该被拒绝
            pass
        
        await engine.close()
        return True, ""
        
    except Exception as e:
        return False, str(e)


async def test_search_functionality() -> Tuple[bool, str]:
    """测试搜索功能"""
    try:
        test_dir = Path(tempfile.mkdtemp()) / "search_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试文档
        test_file = test_dir / "search.md"
        test_file.write_text("搜索功能测试：关键词匹配和语义搜索。")
        
        engine = KnowledgeEngine(
            persist_directory=str(test_dir / "vector_db")
        )
        await engine.add(str(test_file))
        
        # 搜索
        results = await engine.search("搜索功能", top_k=5)
        if not results:
            return False, "搜索没有返回结果"
        
        if not any("搜索" in r.get("content", "") for r in results):
            return False, "搜索结果不相关"
        
        await engine.close()
        shutil.rmtree(test_dir.parent)
        
        return True, ""
        
    except Exception as e:
        return False, str(e)


async def main():
    """运行所有冒烟测试"""
    print_header("KnowledgeCore Engine 冒烟测试")
    
    # 设置测试环境变量
    os.environ.setdefault("DEEPSEEK_API_KEY", "test_key")
    os.environ.setdefault("DASHSCOPE_API_KEY", "test_key")
    
    # 定义测试
    tests = [
        ("基本流程测试", test_basic_flow),
        ("多文件处理测试", test_multiple_files),
        ("错误处理测试", test_error_handling),
        ("搜索功能测试", test_search_functionality),
    ]
    
    # 运行测试
    print(f"{Colors.BOLD}运行测试...{Colors.ENDC}\n")
    
    results = []
    for name, test_func in tests:
        passed, message = await test_func()
        results.append(passed)
        print_test(name, passed, message)
    
    # 总结
    print_header("测试总结")
    
    total = len(results)
    passed = sum(results)
    failed = total - passed
    
    print(f"  总计: {total}")
    print(f"  {Colors.GREEN}通过: {passed}{Colors.ENDC}")
    print(f"  {Colors.RED}失败: {failed}{Colors.ENDC}")
    
    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}所有测试通过！系统运行正常。{Colors.ENDC}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}有测试失败！请检查系统。{Colors.ENDC}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)