"""Place a finished poster image on an exact A4 page without cropping or stretching."""

import argparse
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    image = ImageReader(str(args.image))
    image_width, image_height = image.getSize()
    page_width, page_height = A4
    scale = min(page_width / image_width, page_height / image_height)
    width, height = image_width * scale, image_height * scale
    document = canvas.Canvas(str(args.pdf), pagesize=A4, pageCompression=1)
    document.setTitle("末日物资鉴定局 - A4 宣传海报")
    document.setSubject("废土黑金主题，现实物品与灾后形态对照")
    document.setAuthor("末日物资鉴定局")
    document.setFillColor(HexColor("#080b09"))
    document.rect(0, 0, page_width, page_height, fill=1, stroke=0)
    document.drawImage(
        image, (page_width - width) / 2, (page_height - height) / 2,
        width=width, height=height,
    )
    document.showPage()
    document.save()


if __name__ == "__main__":
    main()
