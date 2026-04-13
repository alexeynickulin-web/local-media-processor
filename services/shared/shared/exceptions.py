"""Shared exceptions for all microservices."""


class MediaProcessorError(Exception):
    """Base exception for all media processor errors."""

    pass


class FileNotFoundError(MediaProcessorError):
    """Raised when a file is not found."""

    pass


class InvalidFileError(MediaProcessorError):
    """Raised when a file is invalid or corrupted."""

    pass


class ModelLoadError(MediaProcessorError):
    """Raised when a model fails to load."""

    pass


class ProcessingError(MediaProcessorError):
    """Raised when processing fails."""

    pass


class TimeoutError(MediaProcessorError):
    """Raised when processing times out."""

    pass


class ValidationError(MediaProcessorError):
    """Raised when input validation fails."""

    pass
