"""文档管理功能的集成测试"""

import pytest
from pathlib import Path
import tempfile
import shutil

from knowledge_core_engine.engine import KnowledgeEngine


class TestDocumentManagementIntegration:
    """文档管理功能的集成测试"""

    @pytest.fixture
    async def engine_with_temp_db(self):
        """创建带临时数据库的引擎"""
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        # 创建引擎
        engine = KnowledgeEngine(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            persist_directory=temp_dir,
            enable_metadata_enhancement=False  # 加快测试速度
        )
        
        yield engine
        
        # 清理
        await engine.close()
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_text_file(self):
        """创建临时文本文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document for KnowledgeCore Engine.\n")
            f.write("It contains sample text for testing document management.\n")
            f.write("The document should be properly indexed and retrievable.")
            temp_path = f.name
        
        yield Path(temp_path)
        
        # 清理
        Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_document_lifecycle(self, engine_with_temp_db, sample_text_file):
        """测试文档的完整生命周期：添加、列出、删除"""
        engine = engine_with_temp_db
        
        # 1. 初始状态应该为空
        initial_list = await engine.list()
        assert initial_list["total"] == 0
        assert initial_list["documents"] == []
        
        # 2. 添加文档
        add_result = await engine.add([sample_text_file])
        assert add_result["processed_files"] == 1
        assert len(add_result["failed_files"]) == 0
        
        # 3. 列出文档，应该有一个
        after_add = await engine.list()
        assert after_add["total"] == 1
        assert len(after_add["documents"]) == 1
        
        # 验证文档信息
        doc = after_add["documents"][0]
        assert doc["name"] == sample_text_file.name
        assert doc["path"] == str(sample_text_file)
        assert doc["chunks_count"] > 0
        assert doc["total_size"] > 0
        
        # 4. 删除文档
        delete_result = await engine.delete(source=sample_text_file.name)
        assert delete_result["deleted_count"] > 0
        
        # 5. 再次列出，应该为空
        after_delete = await engine.list()
        assert after_delete["total"] == 0
        assert after_delete["documents"] == []

    @pytest.mark.asyncio
    async def test_multiple_documents(self, engine_with_temp_db):
        """测试多文档管理"""
        engine = engine_with_temp_db
        
        # 创建多个临时文件
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{i}.txt', delete=False) as f:
                f.write(f"Document {i}: This is test content for document number {i}.\n")
                f.write(f"It has unique content to distinguish it from other documents.\n")
                temp_files.append(Path(f.name))
        
        try:
            # 添加所有文档
            add_result = await engine.add(temp_files)
            assert add_result["processed_files"] == 3
            
            # 列出所有文档
            all_docs = await engine.list()
            assert all_docs["total"] == 3
            
            # 测试分页
            page1 = await engine.list(page=1, page_size=2)
            assert len(page1["documents"]) == 2
            assert page1["pages"] == 2
            
            page2 = await engine.list(page=2, page_size=2)
            assert len(page2["documents"]) == 1
            
            # 删除一个文档
            await engine.delete(source=temp_files[0].name)
            
            # 验证剩余文档
            remaining = await engine.list()
            assert remaining["total"] == 2
            
        finally:
            # 清理临时文件
            for f in temp_files:
                f.unlink()

    @pytest.mark.asyncio
    async def test_document_update(self, engine_with_temp_db, sample_text_file):
        """测试文档更新功能"""
        engine = engine_with_temp_db
        
        # 添加原始文档
        await engine.add([sample_text_file])
        
        # 获取原始文档信息
        original = await engine.list()
        original_size = original["documents"][0]["total_size"]
        
        # 修改文档内容
        with open(sample_text_file, 'a') as f:
            f.write("\nThis is additional content added during update.")
        
        # 更新文档
        update_result = await engine.update(sample_text_file)
        assert update_result["deleted"]["deleted_count"] > 0
        assert update_result["added"]["processed_files"] == 1
        
        # 验证更新后的文档
        updated = await engine.list()
        assert updated["total"] == 1
        assert updated["documents"][0]["total_size"] > original_size

    @pytest.mark.asyncio
    async def test_list_with_stats_toggle(self, engine_with_temp_db, sample_text_file):
        """测试统计信息开关"""
        engine = engine_with_temp_db
        
        # 添加文档
        await engine.add([sample_text_file])
        
        # 带统计信息
        with_stats = await engine.list(return_stats=True)
        assert "chunks_count" in with_stats["documents"][0]
        assert "total_size" in with_stats["documents"][0]
        
        # 不带统计信息
        without_stats = await engine.list(return_stats=False)
        assert "chunks_count" not in without_stats["documents"][0]
        assert "total_size" not in without_stats["documents"][0]
        
        # 但基本信息应该都有
        for result in [with_stats, without_stats]:
            doc = result["documents"][0]
            assert "name" in doc
            assert "path" in doc
            assert "metadata" in doc