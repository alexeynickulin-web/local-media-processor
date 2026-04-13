"""File I/O utilities for all microservices."""

import os
import tempfile
import shutil
from pathlib import Path
from .exceptions import FileNotFoundError, InvalidFileError


ALLOWED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
    ".m4a",
}

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".webm",
    ".mov",
    ".flv",
    ".wmv",
    ".m4v",
}

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tiff",
    ".bmp",
    ".gif",
}

ALLOWED_EXTENSIONS = (
    ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS
)


def validate_file_exists(file_path: str) -> Path:
    """Validate that a file exists.

    Args:
        file_path: Path to the file

    Returns:
        Resolved Path object

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise InvalidFileError(f"Path is not a file: {file_path}")
    return path


def validate_file_extension(file_path: str, allowed_extensions: set[str] | None = None) -> Path:
    """Validate file extension.

    Args:
        file_path: Path to the file
        allowed_extensions: Set of allowed extensions (defaults to all)

    Returns:
        Resolved Path object

    Raises:
        InvalidFileError: If extension not allowed
    """
    path = validate_file_exists(file_path)
    ext = path.suffix.lower()
    allowed = allowed_extensions or ALLOWED_EXTENSIONS

    if ext not in allowed:
        raise InvalidFileError(
            f"File extension '{ext}' not allowed. Allowed: {', '.join(sorted(allowed))}"
        )
    return path


def create_temp_file(suffix: str = "", prefix: str = "media_processor_") -> str:
    """Create a secure temporary file.

    Uses tempfile.mkstemp() to avoid TOCTOU race conditions.

    Args:
        suffix: File suffix/extension
        prefix: File prefix

    Returns:
        Path to the temporary file
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)  # Close the file descriptor, we just need the path
    return path


def safe_remove_file(file_path: str) -> None:
    """Safely remove a file if it exists.

    Args:
        file_path: Path to the file
    """
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
    except OSError:
        pass  # Ignore removal errors


def ensure_directory(directory: str) -> Path:
    """Ensure a directory exists, create if needed.

    Args:
        directory: Path to the directory

    Returns:
        Resolved Path object
    """
    path = Path(directory).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
