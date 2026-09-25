import csv
from pathlib import Path
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

IN_CSV = Path(r"F:\FakeImageDetect\paper_assets\table1_merged\table1_main_auc_percent.csv")
OUT_DOCX = Path(r"F:\FakeImageDetect\paper_assets\table1_merged\table1_main_auc_percent_word_style.docx")

def set_cell_border(cell, top=None, bottom=None, left=None, right=None):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)

    for edge_name, edge_data in {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }.items():
        tag = "w:" + edge_name
        element = tcBorders.find(qn(tag))
        if edge_data is None:
            continue
        if element is None:
            element = OxmlElement(tag)
            tcBorders.append(element)
        for key, value in edge_data.items():
            element.set(qn("w:" + key), str(value))

def clear_cell_borders(cell):
    set_cell_border(
        cell,
        top={"val": "nil"},
        bottom={"val": "nil"},
        left={"val": "nil"},
        right={"val": "nil"},
    )

def set_text(cell, text, bold=False, underline=False, font_size=8.5):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(str(text))
    run.bold = bold
    run.underline = underline
    font = run.font
    font.name = "Times New Roman"
    font.size = Pt(font_size)
    rPr = run._element.rPr
    rFonts = rPr.rFonts
    rFonts.set(qn("w:eastAsia"), "Times New Roman")
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def mark_best_second(rows, headers):
    # 返回 {(row_idx, col_idx): "best"/"second"}
    # row_idx 是数据行下标，从 0 开始；col_idx 是列下标，从 0 开始
    marks = {}
    for j, h in enumerate(headers):
        if j == 0:
            continue
        vals = []
        for i, row in enumerate(rows):
            v = to_float(row[j])
            if v is not None:
                vals.append((v, i))
        if not vals:
            continue
        vals_sorted = sorted(vals, key=lambda x: x[0], reverse=True)
        best_v, best_i = vals_sorted[0]
        marks[(best_i, j)] = "best"

        # 找严格次优，避免并列时乱标
        second = None
        for v, i in vals_sorted[1:]:
            if v < best_v:
                second = (v, i)
                break
        if second is not None:
            marks[(second[1], j)] = "second"
    return marks

with open(IN_CSV, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    data = list(reader)

headers = data[0]
rows = data[1:]

# 你要做成图里那种形式：只保留 Avg.10 或 Avg.25 中更适合正文的 Avg
# 这里默认把 Avg.10 改名为 Avg，把 Avg.25 保留为 Avg.25
# 如果你正文只想放 10 域代表结果，就删除 Avg.25。
keep_headers = []
keep_indices = []
for idx, h in enumerate(headers):
    if h == "Avg.10":
        keep_headers.append("Avg")
        keep_indices.append(idx)
    elif h == "Avg.25":
        # 如果你想保留 25域平均，就取消下面两行注释
        # keep_headers.append("Avg.25")
        # keep_indices.append(idx)
        pass
    else:
        keep_headers.append(h)
        keep_indices.append(idx)

headers2 = keep_headers
rows2 = [[row[i] for i in keep_indices] for row in rows]

marks = mark_best_second(rows2, headers2)

doc = Document()

# 页面横向更适合宽表
section = doc.sections[0]
section.orientation = 1  # landscape
section.page_width, section.page_height = section.page_height, section.page_width
section.top_margin = Pt(36)
section.bottom_margin = Pt(36)
section.left_margin = Pt(36)
section.right_margin = Pt(36)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Table 1. Main comparison on unseen generator domains.")
r.bold = True
r.font.name = "Times New Roman"
r.font.size = Pt(10)
r._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

table = doc.add_table(rows=len(rows2) + 1, cols=len(headers2))
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.autofit = True

# 先清所有边框
for row in table.rows:
    for cell in row.cells:
        clear_cell_borders(cell)

# 表头
for j, h in enumerate(headers2):
    set_text(table.cell(0, j), h, bold=True, font_size=8.5)

# 数据
for i, row in enumerate(rows2):
    for j, val in enumerate(row):
        mark = marks.get((i, j))
        bold = mark == "best"
        underline = mark == "second"
        set_text(table.cell(i + 1, j), val, bold=bold, underline=underline, font_size=8.5)

# 三线表效果：顶线、表头下线、底线
top_border = {"val": "single", "sz": "12", "space": "0", "color": "000000"}
mid_border = {"val": "single", "sz": "6", "space": "0", "color": "000000"}
bottom_border = {"val": "single", "sz": "12", "space": "0", "color": "000000"}

for cell in table.rows[0].cells:
    set_cell_border(cell, top=top_border, bottom=mid_border)

for cell in table.rows[-1].cells:
    set_cell_border(cell, bottom=bottom_border)

# 方法列左对齐
for i in range(len(rows2) + 1):
    cell = table.cell(i, 0)
    for p in cell.paragraphs:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT

note = doc.add_paragraph()
note.alignment = WD_ALIGN_PARAGRAPH.LEFT
run = note.add_run(
    "Note: Results are reported as AUC (%). The best result is shown in bold, "
    "and the second-best result is underlined. Avg denotes the average AUC over the ten representative unseen generator domains."
)
run.font.name = "Times New Roman"
run.font.size = Pt(8)
run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT_DOCX)
print(f"Saved to: {OUT_DOCX}")
