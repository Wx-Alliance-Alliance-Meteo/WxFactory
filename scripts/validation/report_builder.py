
# report_min.py
import os
import io
import base64
import html
from dataclasses import dataclass, field
from typing import List, Optional, Any

import matplotlib.pyplot as plt

def fig_to_png_b64(fig, dpi=140) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def path_to_png_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

class Block:
    def render(self) -> str:
        raise NotImplementedError

@dataclass
class TextBlock(Block):
    text: str
    def render(self) -> str:
        escaped = html.escape(self.text)
        return f'<p style="white-space:pre-line">{escaped}</p>'


@dataclass
class ImageBlock(Block):
    title: Optional[str] = None
    caption: Optional[str] = None
    image_path: Optional[str] = None
    fig: Optional[Any] = None  # matplotlib figure
    alt: str = "image"

    def render(self) -> str:
        if self.fig is not None:
            b64 = fig_to_png_b64(self.fig)
        elif self.image_path is not None:
            b64 = path_to_png_b64(self.image_path)
        else:
            return "<!-- ImageBlock: no image -->"

        title_html = f"<h3>{html.escape(self.title)}</h3>" if self.title else ""
        caption_html = f"<figcaption>{html.escape(self.caption)}</figcaption>" if self.caption else ""
        img_tag = f'<img src="data:image/png;base64,{b64}" alt="{html.escape(self.alt)}"/>'

        return f'<figure class="figure">{title_html}{img_tag}{caption_html}</figure>'


@dataclass
class Section:
    title: Optional[str] = None
    blocks: List[Block] = field(default_factory=list)
    columns: int = 3

    def render(self) -> str:
        title_html = f"<h2>{html.escape(self.title)}</h2>" if self.title else ""
        grid = f'display:grid;grid-template-columns:repeat({max(1,self.columns)},1fr);gap:16px;'
        cells = "\n".join(f'<div class="cell">{b.render()}</div>' for b in self.blocks)
        return f"""{title_html}
    <section class="section" style="{grid}">
    {cells}
    </section>"""

@dataclass
class Report:
    title: str
    sections: List[Section] = field(default_factory=list)

    def css(self) -> str:
        return """
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    h1, h2, h3 { margin: 0 0 12px; }
    .section { margin: 16px 0 28px; }
    .figure img { max-width: 100%; height: auto; border: 1px solid #ddd; }
    .figure figcaption { font-size: 0.9rem; color: #666; margin-top: 6px; }
    .tbl { border-collapse: collapse; width: 100%; }
    .cell { break-inside: avoid; }
    """

    def add(self, section: Section):
        self.sections.append(section)

    def to_html(self) -> str:
        sections_html = "\n".join(s.render() for s in self.sections)
        return f"""<!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8" />
        <title>{html.escape(self.title)}</title>
        <style>{self.css()}</style>
        </head>
        <body>
        <h1>{html.escape(self.title)}</h1>
        {sections_html}
        <footer>End of report?</footer>
        </body>
        </html>"""

    def save(self, path: str = "report.html") -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_html())
        return path
