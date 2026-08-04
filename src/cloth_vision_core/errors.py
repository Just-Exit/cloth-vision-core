class ClothVisionError(Exception):
    """Base error for the public core API."""


class InvalidImageError(ClothVisionError):
    """Raised when an image cannot be safely processed."""


class ProviderError(ClothVisionError):
    """Raised when an external analysis provider fails."""


class InvalidMatchingConfigError(ClothVisionError, ValueError):
    """Raised when a matching JSON configuration is invalid."""
