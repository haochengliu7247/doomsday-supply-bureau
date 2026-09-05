from src.config import Settings
from src.providers.image_edit_base import ImageEditProvider
from src.providers.mock_provider import MockImageEditProvider, MockVLMProvider
from src.providers.vlm_base import VLMProvider


def create_vlm_provider(settings: Settings) -> VLMProvider:
    if settings.mock_mode or settings.vlm_provider == "mock":
        return MockVLMProvider()
    if settings.vlm_provider == "ollama":
        from src.providers.ollama_provider import OllamaProvider

        return OllamaProvider(settings)
    raise ValueError(f"Unsupported VLM provider: {settings.vlm_provider}")


def create_image_provider(settings: Settings) -> ImageEditProvider:
    if settings.mock_mode or settings.image_provider == "mock":
        return MockImageEditProvider(settings.output_dir)
    if settings.image_provider == "comfyui":
        from src.providers.comfyui_provider import ComfyUIProvider

        return ComfyUIProvider(settings)
    raise ValueError(f"Unsupported image provider: {settings.image_provider}")

