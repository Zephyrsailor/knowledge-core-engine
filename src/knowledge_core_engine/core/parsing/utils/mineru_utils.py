"""MinerU document parsing utilities."""

import json
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from knowledge_core_engine.utils.logger import get_logger

logger = get_logger(__name__)


class MineruUtils:
    """Utility class for MinerU operations."""
    
    @staticmethod
    def parse_pdf(
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        source: str = "local",
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Parse PDF document using MinerU.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory path
            method: Parsing method (auto, txt, ocr)
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for mineru command
            
        Returns:
            Tuple[List[Dict[str, Any]], str]: Tuple containing (content list JSON, Markdown text)
        """
        try:
            # Convert to Path object for easier handling
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

            name_without_suff = pdf_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = pdf_path.parent / "mineru_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Run mineru command
            MineruUtils._run_mineru_command(
                input_path=pdf_path,
                output_dir=base_output_dir,
                method=method,
                lang=lang,
                source=source,
                **kwargs,
            )

            # Read the generated output files
            backend = kwargs.get("backend", "")
            if backend.startswith("vlm-"):
                method = "vlm"

            content_list, md_content = MineruUtils._read_output_files(
                base_output_dir, name_without_suff, method=method
            )

            return content_list, md_content

        except Exception as e:
            logger.error(f"Error in parse_pdf: {str(e)}")
            raise
    
    @staticmethod
    def _run_mineru_command(
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        method: str = "auto",
        lang: Optional[str] = None,
        backend: str = "pipeline",
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        formula: bool = True,
        table: bool = True,
        device: Optional[str] = None,
        source: str = "huggingface",
        vlm_url: Optional[str] = None,
    ) -> None:
        """Run mineru command line tool."""
        cmd = [
            "mineru",
            "-p",
            str(input_path),
            "-o",
            str(output_dir),
            "-m",
            method,
            "-b",
            backend,
            "--source",
            source,
        ]

        if lang:
            cmd.extend(["-l", lang])
        if start_page is not None:
            cmd.extend(["-s", str(start_page)])
        if end_page is not None:
            cmd.extend(["-e", str(end_page)])
        if not formula:
            cmd.extend(["-f", "false"])
        if not table:
            cmd.extend(["-t", "false"])
        if device:
            cmd.extend(["-d", device])
        if vlm_url:
            cmd.extend(["-u", vlm_url])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                errors="ignore",
            )
            logger.info("MinerU command executed successfully")
            if result.stdout:
                logger.debug(f"MinerU output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running mineru command: {e}")
            if e.stderr:
                logger.error(f"Error details: {e.stderr}")
            raise
        except FileNotFoundError:
            raise RuntimeError(
                "mineru command not found. Please ensure MinerU 2.0 is properly installed:\n"
                "pip install -U 'mineru[core]' or uv pip install -U 'mineru[core]'"
            )
    
    @staticmethod
    def _read_output_files(
        output_dir: Path, file_stem: str, method: str = "auto"
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Read the output files generated by mineru."""
        # Look for the generated files
        md_file = output_dir / f"{file_stem}.md"
        json_file = output_dir / f"{file_stem}_content_list.json"

        # Check for files in subdirectory (MinerU 2.0 may create subdirectories)
        subdir = output_dir / file_stem
        if subdir.exists():
            md_file = subdir / method / f"{file_stem}.md"
            json_file = subdir / method / f"{file_stem}_content_list.json"

        # Read markdown content
        md_content = ""
        if md_file.exists():
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    md_content = f.read()
            except Exception as e:
                logger.warning(f"Could not read markdown file {md_file}: {e}")

        # Read JSON content list
        content_list = []
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    content_list = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read JSON file {json_file}: {e}")

        return content_list, md_content
    
    @staticmethod
    def parse_image(
        image_path: Path,
        output_dir: Path,
        method: str = "auto",
        lang: str = "ch",
        backend: str = "pipeline",
        source: str = "local",
        device: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Parse image using MinerU OCR.
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to store output files
            method: Parsing method
            lang: Language for OCR
            backend: Backend to use
            source: Source type
            device: Device to use for processing
            
        Returns:
            Tuple of (content_list, markdown_content)
        """
        # Supported image formats by MinerU
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        unsupported_formats = {'.gif', '.webp', '.svg'}
        
        file_stem = image_path.stem
        file_suffix = image_path.suffix.lower()
        
        # Convert unsupported formats to PNG
        if file_suffix in unsupported_formats:
            logger.info(f"Converting {file_suffix} to PNG for MinerU processing")
            temp_dir = Path(tempfile.mkdtemp())
            try:
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    
                    converted_path = temp_dir / f"{file_stem}.png"
                    img.save(converted_path, 'PNG')
                    
                    # Parse the converted image
                    return MineruUtils._parse_image_with_mineru(
                        converted_path, output_dir, method, lang, backend, source, device
                    )
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        elif file_suffix in supported_formats:
            return MineruUtils._parse_image_with_mineru(
                image_path, output_dir, method, lang, backend, source, device
            )
        else:
            raise ValueError(f"Unsupported image format: {file_suffix}")
    
    @staticmethod
    def _parse_image_with_mineru(
        image_path: Path,
        output_dir: Path,
        method: str,
        lang: str,
        backend: str,
        source: str,
        device: Optional[str]
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Parse image using MinerU OCR method."""
        try:
            # Run MinerU OCR command
            cmd = [
                "mineru", "ocr",
                "--method", method,
                "--lang", lang,
                "--backend", backend,
                "--source", source,
                "--output-dir", str(output_dir),
                str(image_path)
            ]
            
            if device:
                cmd.extend(["--device", device])
            
            logger.info(f"Running MinerU OCR command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minutes timeout
            )
            
            logger.info(f"MinerU OCR completed successfully for {image_path.name}")
            
            # Read output files
            return MineruUtils._read_output_files(output_dir, image_path.stem, method)
            
        except subprocess.TimeoutExpired:
            logger.error(f"MinerU OCR timed out for {image_path}")
            raise Exception("MinerU OCR processing timed out")
        except subprocess.CalledProcessError as e:
            logger.error(f"MinerU OCR failed for {image_path}: {e.stderr}")
            raise Exception(f"MinerU OCR failed: {e.stderr}")
        except FileNotFoundError:
            logger.error("MinerU command not found. Please ensure MinerU is installed and in PATH.")
            raise Exception("MinerU not found. Please install MinerU.")
    
    @staticmethod
    def parse_office_doc(
        doc_path: Path,
        output_dir: Path,
        method: str = "auto",
        lang: str = "ch",
        backend: str = "pipeline",
        source: str = "local",
        formula: bool = True,
        table: bool = True,
        device: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Parse Office document by converting to PDF first.
        
        Args:
            doc_path: Path to the Office document
            output_dir: Directory to store output files
            method: Parsing method
            lang: Language for parsing
            backend: Backend to use
            source: Source type
            formula: Whether to extract formulas
            table: Whether to extract tables
            device: Device to use for processing
            
        Returns:
            Tuple of (content_list, markdown_content)
        """
        # Check if LibreOffice is available
        # libreoffice_commands = [ 'soffice', 'libreoffice', 'loffice']
        # libreoffice_cmd = None

        """检查LibreOffice是否可用"""
        if not shutil.which("soffice"):
            logger.warning("shutil.which检测LibreOffice未安装或不可用")
            raise Exception("shutil.which检测LibreOffice未安装或不可用")


        
        # Create temporary directory for PDF conversion
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Convert Office document to PDF
            logger.info(f"Converting {doc_path.name} to PDF using LibreOffice")
            
            conversion_cmd = [
                'soffice',
                '--headless',
                '--convert-to', 'pdf',
                '--outdir', str(temp_dir),
                str(doc_path)
            ]
            
            result = subprocess.run(
                conversion_cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout for conversion
            )
            
            if result.returncode != 0:
                logger.error(f"LibreOffice conversion failed: {result.stderr}")
                raise Exception(f"Failed to convert {doc_path.name} to PDF: {result.stderr}")
            
            # Find the generated PDF
            pdf_path = temp_dir / f"{doc_path.stem}.pdf"
            if not pdf_path.exists():
                # Sometimes LibreOffice might change the filename
                pdf_files = list(temp_dir.glob("*.pdf"))
                if pdf_files:
                    pdf_path = pdf_files[0]
                else:
                    raise Exception(f"PDF conversion failed - no PDF file generated")
            
            logger.info(f"Successfully converted {doc_path.name} to PDF")
            
            # Parse the generated PDF using MinerU
            return MineruUtils.parse_pdf(
                pdf_path=pdf_path,
                output_dir=output_dir,
                method=method,
                lang=lang,
                backend=backend,
                source=source,
                formula=formula,
                table=table,
                device=device
            )
            
        except subprocess.TimeoutExpired:
            logger.error(f"LibreOffice conversion timed out for {doc_path}")
            raise Exception("Office document conversion timed out")
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @staticmethod
    def parse_text_file(
        text_path: Path,
        output_dir: Path,
        method: str = "auto",
        lang: str = "ch",
        backend: str = "pipeline",
        source: str = "local",
        formula: bool = True,
        table: bool = True,
        device: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Parse text file by converting to PDF first.
        
        Args:
            text_path: Path to the text file
            output_dir: Directory to store output files
            method: Parsing method
            lang: Language for parsing
            backend: Backend to use
            source: Source type
            formula: Whether to extract formulas
            table: Whether to extract tables
            device: Device to use for processing
            
        Returns:
            Tuple of (content_list, markdown_content)
        """
        # Read text content with encoding detection
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
        text_content = None
        
        for encoding in encodings:
            try:
                with open(text_path, 'r', encoding=encoding) as f:
                    text_content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if text_content is None:
            raise Exception(f"Could not decode text file {text_path} with any supported encoding")
        
        # Create temporary directory for PDF conversion
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Convert text to PDF using reportlab
            pdf_path = temp_dir / f"{text_path.stem}.pdf"
            
            logger.info(f"Converting {text_path.name} to PDF")
            
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Split text into paragraphs and add to PDF
            paragraphs = text_content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    p = Paragraph(para.replace('\n', '<br/>'), styles['Normal'])
                    story.append(p)
                    story.append(Spacer(1, 12))
            
            doc.build(story)
            
            logger.info(f"Successfully converted {text_path.name} to PDF")
            
            # Parse the generated PDF using MinerU
            return MineruUtils.parse_pdf(
                pdf_path=pdf_path,
                output_dir=output_dir,
                method=method,
                lang=lang,
                backend=backend,
                source=source,
                formula=formula,
                table=table,
                device=device
            )
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

class Qwen25VL72BInstruct:
    def __init__(self):
        self.model = "Qwen/Qwen2.5-VL-72B-Instruct"
        self.api_key = os.getenv("MODELSCOPE_API_KEY")
        self.api_base = "https://api-inference.modelscope.cn/v1/"