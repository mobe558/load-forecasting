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
:root { --navy: #0f2a4a; --blue: #2a78d6; --teal: #1baf7a; --orange: #eb6834; --ink: #1c2430;
  --muted: #5b6573; --line: #e3e8ef; --soft: #f3f6fa; }
* { box-sizing: border-box; }
body { margin: 0; font-family: 'Avenir Next', 'Helvetica Neue', Arial, sans-serif; color: var(--ink);
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
section.slide { width: 297mm; height: 167mm; padding: 11mm 16mm 13mm; page-break-after: always; overflow: hidden;
  position: relative; display: flex; flex-direction: column; background: #fff; }
.kicker { font-size: 8.5pt; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--blue);
  margin: 0 0 1.5mm; }
section.slide h2 { font-size: 21pt; font-weight: 700; color: var(--navy); margin: 0 0 6mm; line-height: 1.15; }
section.slide h2::after { content: ""; display: block; width: 16mm; height: 1.1mm; background: var(--blue);
  border-radius: 1mm; margin-top: 2.5mm; }
section.slide p, section.slide li { font-size: 11.5pt; line-height: 1.42; margin: 0; }
section.slide ul { margin: 0; padding-left: 5mm; } section.slide li { margin-bottom: 1.6mm; }
section.slide li::marker { color: var(--blue); }
b, strong { color: var(--navy); }
.body { flex: 1; min-height: 0; display: flex; flex-direction: column; gap: 5mm; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 9mm; min-height: 0; }
.cols.grow { flex: 1; } .body > .takeaway { margin-top: auto; }
.fig-s img { max-height: 60mm; } .fig-m img { max-height: 74mm; }
.cols.wide-left { grid-template-columns: 1.35fr 1fr; } .cols.wide-right { grid-template-columns: 1fr 1.35fr; }
.cols.three { grid-template-columns: repeat(3, 1fr); gap: 5mm; }
.card { background: var(--soft); border-radius: 2.5mm; padding: 4mm 5mm; border-top: 1.2mm solid var(--blue); }
.card.teal { border-top-color: var(--teal); } .card.orange { border-top-color: var(--orange); }
.card.grey { border-top-color: #8a94a3; } .card.navy { border-top-color: var(--navy); }
.card h3 { font-size: 11.5pt; margin: 0 0 1.5mm; color: var(--navy); }
.card .tag { font-size: 7.5pt; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
.card p, .card li { font-size: 10pt; line-height: 1.38; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 5mm; }
.kpi { background: var(--soft); border-radius: 2.5mm; padding: 3.5mm 4.5mm; }
.kpi .num { font-size: 22pt; font-weight: 700; color: var(--navy); line-height: 1.1; }
.kpi .num.blue { color: var(--blue); } .kpi .num.teal { color: var(--teal); } .kpi .num.orange { color: var(--orange); }
.kpi .lbl { font-size: 8.8pt; color: var(--muted); margin-top: 1mm; line-height: 1.3; }
table.t { width: 100%; border-collapse: collapse; font-size: 9.6pt; }
table.t th { background: var(--navy); color: #fff; font-weight: 600; text-align: left; padding: 1.8mm 2.6mm; }
table.t td { padding: 1.6mm 2.6mm; border-bottom: 1px solid var(--line); }
table.t td.n, table.t th.n { text-align: right; font-variant-numeric: tabular-nums; }
table.t tr.best td { background: #e7f0fc; font-weight: 700; color: var(--navy); }
table.t tr.base td { color: var(--muted); }
.badge { display: inline-block; padding: .5mm 2.4mm; border-radius: 8mm; font-size: 8.2pt; font-weight: 700; white-space: nowrap; }
.ok { background: #dff5ea; color: #11744f; } .mid { background: #fff1d9; color: #9a5b06; } .no { background: #fde4e1; color: #a3291c; }
.takeaway { background: var(--navy); color: #fff; border-radius: 2.5mm; padding: 3mm 5mm; font-size: 11pt; line-height: 1.4; }
.takeaway b { color: #9cc6ff; }
figure { margin: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 0; }
figure img { max-width: 100%; max-height: 82mm; object-fit: contain; }
figcaption { font-size: 8.5pt; color: var(--muted); margin-top: 1.5mm; text-align: center; }
.flow { display: flex; align-items: center; gap: 2.2mm; }
.flow .box { flex: 1; background: var(--soft); border: 1px solid var(--line); border-radius: 2mm; padding: 2.5mm 2.5mm;
  text-align: center; font-size: 9pt; line-height: 1.3; }
.flow .box b { display: block; font-size: 9.8pt; margin-bottom: .6mm; }
.flow .box.hl { background: #e7f0fc; border-color: #b9d3f5; } .flow .box.out { background: #e3f6ee; border-color: #b4e3cd; }
.flow .arrow { color: var(--blue); font-weight: 700; font-size: 13pt; }
.lane { font-size: 8pt; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin: 0 0 1.5mm; }
.small { font-size: 9.5pt !important; color: var(--muted); }
.foot { position: absolute; left: 16mm; right: 16mm; bottom: 5mm; display: flex; justify-content: space-between;
  font-size: 7.5pt; color: #8a94a3; border-top: 1px solid var(--line); padding-top: 1.8mm; }
section.dark { background: linear-gradient(135deg, #0b2140 0%, #123a6b 60%, #1d5aa6 100%); color: #fff; padding: 16mm 20mm; }
section.dark b, section.dark strong { color: #fff; }
.title-wrap { display: flex; flex-direction: column; height: 100%; }
.title-wrap .eyebrow { font-size: 9pt; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; color: #9cc6ff; }
.title-wrap h1 { font-size: 34pt; line-height: 1.1; margin: 4mm 0 3mm; color: #fff; font-weight: 700; }
.title-wrap .sub { font-size: 14pt; color: #d6e6fb; margin: 0; }
.title-wrap .bar { width: 22mm; height: 1.4mm; background: #5fb0ff; border-radius: 1mm; margin: 7mm 0 6mm; }
.team { display: grid; grid-template-columns: auto 1fr; gap: 1.2mm 6mm; font-size: 10pt; color: #e8f1fd; }
.team .id { color: #9cc6ff; font-variant-numeric: tabular-nums; }
.team-name { font-size: 11pt; font-weight: 700; color: #fff; margin-bottom: 2.5mm; letter-spacing: .03em; }
.title-wrap .meta { margin-top: auto; font-size: 9pt; color: #b7cdea; }
.closing { display: flex; flex-direction: column; justify-content: center; height: 100%; }
.closing h1 { font-size: 40pt; margin: 0 0 4mm; color: #fff; }
.closing p { font-size: 14pt; color: #d6e6fb; }
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
    """slides.md: slides separated by a line '---'. A slide may start with '<!-- slide: dark -->' to set
    extra CSS classes. Markdown and raw HTML are both allowed; {{table:..}} / {{fig:..}} placeholders work."""
    import markdown
    parts = [p.strip() for p in re.split(r"^---\s*$", (REP / "slides.md").read_text(), flags=re.M) if p.strip()]
    secs = []
    for i, p in enumerate(parts, 1):
        m = re.match(r"<!--\s*slide:\s*([\w\s-]*)-->", p)
        cls = "slide " + (m.group(1).strip() if m else "")
        body = markdown.markdown(fill(p), extensions=["tables", "md_in_html", "attr_list"])
        foot = "" if "dark" in cls else (f'<div class="foot"><span>{team} · Multi-horizon Electricity Load '
                                         f'Forecasting · AI403</span><span>{i} / {len(parts)}</span></div>')
        secs.append(f'<section class="{cls.strip()}">{body}{foot}</section>')
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
