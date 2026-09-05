import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from src.config import get_settings
from src.logging_config import configure_logging
from src.ui.gradio_app import (
    allowed_asset_paths,
    build_app,
    build_theme,
    reveal_script,
    stylesheet_path,
)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    demo = build_app(settings)
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=settings.app_host,
        server_port=settings.app_port,
        share=settings.app_share,
        show_error=False,
        allowed_paths=allowed_asset_paths(settings),
        theme=build_theme(),
        css_paths=str(stylesheet_path()),
        js=reveal_script(),
        max_file_size=f"{settings.max_upload_mb}mb",
        footer_links=[],
    )


if __name__ == "__main__":
    main()
