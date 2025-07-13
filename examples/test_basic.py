"""Test basic functionality."""

import asyncio
from pathlib import Path
from knowledge_core_engine.core.parsing.document_processor import DocumentProcessor
from knowledge_core_engine.core.chunking.pipeline import ChunkingPipeline


async def main():
    print("Testing basic document processing...")
    
    # Create test document
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document. It contains information about RAG technology.")
        test_file = Path(f.name)
    
    try:
        # Test parsing
        parser = DocumentProcessor()
        result = await parser.process(test_file)
        print(f"✅ Parsing: {len(result.markdown)} characters")
        
        # Test chunking
        chunker = ChunkingPipeline()
        chunks = await chunker.process_parse_result(result)
        print(f"✅ Chunking: {chunks.total_chunks} chunks created")
        
        print("\n✅ All basic tests passed!")
        
    finally:
        test_file.unlink()


if __name__ == "__main__":
    asyncio.run(main())