"""Build report/forecast_AI_Report.pdf from report/report.md and report/forecast_AI_Slides.pdf from report/slides.md.

  python report/build_report.py forecast_AI          # report + slides
  python report/build_report.py forecast_AI --slides # slides only

report.md may contain placeholders that are filled from the experiment outputs, so the report always
shows the numbers that the code produced:
  {{table:<stem>}}   -> experiments/tables/<stem>.md (without its heading)
  {{fig:<path>|<caption>}} -> an image (path relative to the repository root) with a caption
Markdown -> HTML with pandoc if installed, else the `markdown` Python package; the PDF is printed with
headless Chrome/Chromium/Edge/Brave (falls back to wkhtmltopdf).
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "report"

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 20pt; margin-top: 0; } h2 { font-size: 14pt; border-bottom: 1px solid #ccc; padding-bottom: 3px;
 margin-top: 22px; page-break-after: avoid; } h3 { font-size: 11.5pt; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 8.3pt; margin: 8px 0 14px; page-break-inside: avoid; }
th, td { border: 1px solid #d0d0d0; padding: 3px 5px; text-align: left; vertical-align: top; }
th { background: #f1f1ef; } tr:nth-child(even) td { background: #fafaf9; }
figure { margin: 10px 0 16px; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 9.5cm; } figcaption { font-size: 9pt; color: #555; margin-top: 4px; }
code { font-size: 9pt; background: #f4f4f2; padding: 0 2px; } pre { font-size: 8.5pt; background: #f4f4f2; padding: 6px;
 white-space: pre-wrap; page-break-inside: avoid; }
.title { text-align: center; margin-top: 60px; } .title h1 { font-size: 24pt; } .title table { width: auto; margin: 10px auto 16px; font-size: 10pt; } .pb { page-break-after: always; }
blockquote { border-left: 3px solid #2a78d6; margin: 8px 0; padding: 2px 10px; color: #333; background: #f5f8fd; }
"""


def fill(md: str) -> str:
    def table(m):
        p = ROOT / "experiments" / "tables" / f"{m.group(1)}.md"
        if not p.exists():
            return f"*[missing table {m.group(1)} - run src/evaluate.py]*"
        return "\n".join(line for line in p.read_text().splitlines() if not line.startswith("### "))

    def fig(m):
        path, cap = m.group(1).split("|", 1)
        return f'<figure><img src="{(ROOT / path).as_uri()}"><figcaption>{cap}</figcaption></figure>'

    md = re.sub(r"\{\{table:([\w\-]+)\}\}", table, md)
    return re.sub(r"\{\{fig:([^}]+)\}\}", fig, md)


SLIDE_CSS = """
@page { size: 297mm 167mm; margin: 0; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; color: #1a1a1a; }
section.slide { width: 297mm; height: 167mm; box-sizing: border-box; padding: 11mm 15mm 9mm; page-break-after: always;
 overflow: hidden; position: relative; display: flex; flex-direction: column; }
section.slide h1 { font-size: 26pt; margin: 0 0 6mm; color: #12355b; }
section.slide h2 { font-size: 19pt; margin: 0 0 4mm; color: #12355b; border-bottom: 2px solid #2a78d6; padding-bottom: 2mm; }
section.slide p, section.slide li { font-size: 12.5pt; line-height: 1.38; } section.slide ul { margin: 1mm 0; padding-left: 6mm; }
section.slide table { border-collapse: collapse; font-size: 9.5pt; margin: 2mm 0; }
section.slide th, section.slide td { border: 1px solid #d0d0d0; padding: 1.5mm 2.5mm; text-align: left; }
section.slide th { background: #eef3fa; }
section.slide figure { margin: 1mm 0; text-align: center; flex: 1; min-height: 0; display: flex; flex-direction: column;
 align-items: center; justify-content: center; }
section.slide figure img { max-width: 100%; max-height: 100mm; } section.slide figcaption { font-size: 9pt; color: #555; }
.cols { display: flex; gap: 8mm; } .cols > div { flex: 1; min-width: 0; }
.foot { position: absolute; bottom: 4mm; right: 15mm; font-size: 8pt; color: #888; }
.title-slide { justify-content: center; text-align: center; } .title-slide h1 { font-size: 32pt; }
"""


def md_to_html(md_path: Path, css_path: Path, html: Path):
    if shutil.which("pandoc"):
        subprocess.run(["pandoc", str(md_path), "-f", "markdown+pipe_tables+raw_html", "-s",
                        "--metadata", "title=report", "-c", str(css_path), "-o", str(html)], check=True)
        h = html.read_text()
        h = re.sub(r'<header id="title-block-header">.*?</header>', "", h, flags=re.S)   # drop pandoc's title
    else:
        import markdown
        body = markdown.markdown(md_path.read_text(), extensions=["tables", "fenced_code", "md_in_html", "attr_list"])
        h = (f'<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="{css_path.as_uri()}">'
             f"</head><body>{body}</body></html>")
    html.write_text(h)


def find_chrome():
    for c in ["google-chrome", "chromium", "chromium-browser", "microsoft-edge", "brave-browser"]:
        if shutil.which(c):
            return shutil.which(c)
    for app in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium",
                "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
                r"C:\Program Files\Google\Chrome\Application\chrome.exe"]:
        if Path(app).exists():
            return app
    return None


def to_pdf(html: Path, pdf: Path):
    chrome = find_chrome()
    if chrome:
        subprocess.run([chrome, "--headless", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                        "--allow-file-access-from-files", f"--print-to-pdf={pdf}", html.as_uri()],
                       check=True, capture_output=True)
    else:
        subprocess.run(["wkhtmltopdf", "--enable-local-file-access", str(html), str(pdf)], check=True)
    print(f"wrote {pdf}")


def build_report(team):
    md = fill((REP / "report.md").read_text())
    (REP / "_filled.md").write_text(md)
    (REP / "_style.css").write_text(CSS)
    html = REP / "_report.html"
    md_to_html(REP / "_filled.md", REP / "_style.css", html)
    to_pdf(html, REP / f"{team}_Report.pdf")


def build_slides(team):
    """slides.md: slides separated by a line '---'; each slide is Markdown (raw HTML allowed)."""
    import markdown
    parts = [p.strip() for p in re.split(r"^---\s*$", (REP / "slides.md").read_text(), flags=re.M) if p.strip()]
    secs = []
    for i, p in enumerate(parts, 1):
        cls = "slide title-slide" if p.startswith("<!-- title -->") else "slide"
        body = markdown.markdown(fill(p), extensions=["tables", "md_in_html", "attr_list"])
        foot = "" if i == 1 else f'<div class="foot">{team} · {i}/{len(parts)}</div>'
        secs.append(f'<section class="{cls}" markdown="1">{body}{foot}</section>')
    html = REP / "_slides.html"
    html.write_text(f'<!doctype html><html><head><meta charset="utf-8"><style>{SLIDE_CSS}</style></head>'
                    f'<body>{"".join(secs)}</body></html>')
    to_pdf(html, REP / f"{team}_Slides.pdf")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    team = args[0] if args else "forecast_AI"
    if "--slides" not in sys.argv:
        build_report(team)
    if (REP / "slides.md").exists():
        build_slides(team)


if __name__ == "__main__":
    main()
