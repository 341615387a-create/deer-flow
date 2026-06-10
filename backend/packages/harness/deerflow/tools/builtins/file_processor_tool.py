import os
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_enable_pir'] = '0'
os.environ['FLAGS_new_executor'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_ONEDNN_ENABLE'] = '0'
os.environ['PADDLE_PIR_ENABLE'] = '0'

from pathlib import Path
from typing import Annotated

from langchain.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deerflow.agents.thread_state import ThreadDataState
from deerflow.config.paths import VIRTUAL_PATH_PREFIX
from deerflow.sandbox.exceptions import SandboxRuntimeError
from deerflow.sandbox.tools import (
    get_thread_data,
    resolve_and_validate_user_data_path,
    validate_local_tool_path,
)
from deerflow.tools.types import Runtime


_ALLOWED_EXTENSIONS = {
    'image': {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'},
    'document': {'.docx'},
}

_paddle_ocr_instance = None


def _get_paddle_ocr():
    """获取 PaddleOCR 实例（单例模式）"""
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        try:
            # DISABLED: from paddleocr import PaddleOCR
            _paddle_ocr_instance = PaddleOCR(lang='ch')
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PaddleOCR: {str(e)}")
    return _paddle_ocr_instance


def _sanitize_error(error: Exception, thread_data: ThreadDataState | None) -> str:
    from deerflow.sandbox.tools import mask_local_paths_in_output
    return mask_local_paths_in_output(f"{type(error).__name__}: {error}", thread_data)


def _extract_text_with_paddleocr(image_path: str) -> str:
    """
    使用 PaddleOCR 从图片中提取文字（支持中文和英文）
    
    Args:
        image_path: 图片文件的绝对路径
    
    Returns:
        提取的文字内容
    """
    try:
        # DISABLED: from paddleocr import PaddleOCR
        
        # DISABLED: ocr = PaddleOCR(lang='ch', use_angle_cls=False)
        result = ocr.ocr(image_path)
        
        if not result or not result[0]:
            return "未识别到文字内容"
        
        text_lines = []
        for line in result[0]:
            if line[1] and line[1][0]:
                text_lines.append(line[1][0])
        
        return '\n'.join(text_lines)
    except ImportError:
        return "错误: PaddleOCR 未安装"
    except Exception as e:
        return f"OCR识别失败: {str(e)}"


def _extract_text_with_tesseract(image_path: str) -> str:
    """
    使用 Tesseract OCR 从图片中提取文字（备用方案）
    
    Args:
        image_path: 图片文件的绝对路径
    
    Returns:
        提取的文字内容
    """
    try:
        import pytesseract
        from PIL import Image
        
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='chi_sim+eng')
        
        if not text.strip():
            return "未识别到文字内容"
        
        return text
    except ImportError as e:
        return f"错误: pytesseract 未安装: {str(e)}"
    except Exception as e:
        return f"OCR识别失败: {str(e)}"


@tool("extract_text_from_image", parse_docstring=True)
def extract_text_from_image_tool(
    runtime: Runtime,
    image_path: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Extract text from an image using OCR (Optical Character Recognition).

    Supports Chinese and English text extraction using PaddleOCR.

    Args:
        image_path: Absolute /mnt/user-data virtual path to the image file. Supported formats: jpg, jpeg, png, webp, bmp, tiff.
    """
    thread_data = get_thread_data(runtime)
    requested_path = image_path

    if not image_path.startswith(VIRTUAL_PATH_PREFIX):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Error: Image path must be under {VIRTUAL_PATH_PREFIX}",
                        tool_call_id=tool_call_id,
                    )
                ]
            },
        )

    try:
        validate_local_tool_path(image_path, thread_data, read_only=True)
        actual_path = resolve_and_validate_user_data_path(image_path, thread_data)
    except (PermissionError, SandboxRuntimeError) as e:
        return Command(
            update={"messages": [ToolMessage(f"Error: {str(e)}", tool_call_id=tool_call_id)]},
        )

    path = Path(actual_path)

    if not path.exists():
        return Command(
            update={"messages": [ToolMessage(f"Error: Image file not found: {image_path}", tool_call_id=tool_call_id)]},
        )

    if not path.is_file():
        return Command(
            update={"messages": [ToolMessage(f"Error: Path is not a file: {image_path}", tool_call_id=tool_call_id)]},
        )

    ext = path.suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS['image']:
        return Command(
            update={"messages": [ToolMessage(f"Error: Unsupported image format: {ext}. Supported formats: {', '.join(_ALLOWED_EXTENSIONS['image'])}", tool_call_id=tool_call_id)]},
        )

    try:
        text = _extract_text_with_tesseract(actual_path)
        
        if text.startswith("错误:"):
            text = _extract_text_with_paddleocr(actual_path)
        
        return Command(
            update={"messages": [ToolMessage(text, tool_call_id=tool_call_id)]},
        )
    except Exception as e:
        return Command(
            update={"messages": [ToolMessage(f"Error extracting text: {_sanitize_error(e, thread_data)}", tool_call_id=tool_call_id)]},
        )


@tool("read_docx", parse_docstring=True)
def read_docx_tool(
    runtime: Runtime,
    file_path: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Read the contents of a DOCX document.

    Args:
        file_path: Absolute /mnt/user-data virtual path to the DOCX file.
    """
    thread_data = get_thread_data(runtime)
    requested_path = file_path

    if not file_path.startswith(VIRTUAL_PATH_PREFIX):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Error: File path must be under {VIRTUAL_PATH_PREFIX}",
                        tool_call_id=tool_call_id,
                    )
                ]
            },
        )

    try:
        validate_local_tool_path(file_path, thread_data, read_only=True)
        actual_path = resolve_and_validate_user_data_path(file_path, thread_data)
    except (PermissionError, SandboxRuntimeError) as e:
        return Command(
            update={"messages": [ToolMessage(f"Error: {str(e)}", tool_call_id=tool_call_id)]},
        )

    path = Path(actual_path)

    if not path.exists():
        return Command(
            update={"messages": [ToolMessage(f"Error: File not found: {file_path}", tool_call_id=tool_call_id)]},
        )

    if not path.is_file():
        return Command(
            update={"messages": [ToolMessage(f"Error: Path is not a file: {file_path}", tool_call_id=tool_call_id)]},
        )

    ext = path.suffix.lower()
    if ext != '.docx':
        return Command(
            update={"messages": [ToolMessage(f"Error: Unsupported format: {ext}. Only DOCX files are supported.", tool_call_id=tool_call_id)]},
        )

    try:
        from docx import Document
        
        doc = Document(actual_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        
        if not full_text:
            text = "文档内容为空"
        else:
            text = '\n'.join(full_text)
        
        return Command(
            update={"messages": [ToolMessage(text, tool_call_id=tool_call_id)]},
        )
    except ImportError:
        return Command(
            update={"messages": [ToolMessage("Error: python-docx not installed", tool_call_id=tool_call_id)]},
        )
    except Exception as e:
        return Command(
            update={"messages": [ToolMessage(f"Error reading document: {_sanitize_error(e, thread_data)}", tool_call_id=tool_call_id)]},
        )