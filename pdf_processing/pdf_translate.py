import os
import json
import math
import logging
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from spire.pdf import PdfDocument, PdfImageHelper
from camelot.io import read_pdf
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from uuid import uuid4
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from PIL import Image
import warnings
import gc
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet

warnings.filterwarnings("ignore")

import warnings as _warnings
_warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*ARC4.*")

# --- Rectangle extraction logic ---
def extract_rects(input_pdf_path: str) -> List[Dict[str, Any]]:
    results = []
    doc = fitz.open(input_pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        paths = page.get_drawings()
        for path in paths:
            items = path["items"]
            for item in items:
                if item[0] == "re":
                    if len(item) == 3 and isinstance(item[1], fitz.Rect):
                        rect = item[1]
                        results.append({
                            "type": "rect",
                            "page": page_num,
                            "x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1
                        })
                    elif len(item) >= 5 and all(isinstance(val, (int, float)) for val in item[1:5]):
                        x, y, w, h = item[1], item[2], item[3], item[4]
                        results.append({
                            "type": "rect",
                            "page": page_num,
                            "x0": x, "y0": y, "x1": x + w, "y1": y + h
                        })
    doc.close()
    return results

# --- Helper functions for box logic ---
def rect_nearby_any_side(r1: fitz.Rect, r2: fitz.Rect, threshold: float = 30) -> bool:
    return (
        abs(r1.x0 - r2.x0) <= threshold or
        abs(r1.y0 - r2.y0) <= threshold or
        abs(r1.x1 - r2.x1) <= threshold or
        abs(r1.y1 - r2.y1) <= threshold
    )

def rect_fully_contained(inner: fitz.Rect, outer: fitz.Rect) -> bool:
    return (
        inner.x0 >= outer.x0 and inner.y0 >= outer.y0 and
        inner.x1 <= outer.x1 and inner.y1 <= outer.y1
    )

def rect_iou(r1: fitz.Rect, r2: fitz.Rect) -> float:
    inter = r1 & r2
    if inter.is_empty:
        return 0.0
    inter_area = inter.width * inter.height
    union_area = r1.width * r1.height + r2.width * r2.height - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

# --- Hyperlink extraction from original PDF ---
def extract_hyperlinks(input_pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(input_pdf_path)
    links = []
    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
        except AttributeError:
            page = doc[page_num]
        get_links_fn = getattr(page, "get_links", None)
        if get_links_fn is None:
            get_links_fn = getattr(page, "getLinks", lambda: [])
        page_links = get_links_fn() if callable(get_links_fn) else []
        if not isinstance(page_links, list):
            page_links = []
        for link in page_links:
            kind_uri = getattr(fitz, "LINK_URI", 2)
            if link.get("kind", None) == kind_uri and link.get("uri"):
                bbox = fitz.Rect(link["from"])
                text = ""
                get_text_fn = getattr(page, "get_text", None)
                if get_text_fn is None:
                    get_text_fn = getattr(page, "getText", lambda _: {"blocks": []})
                blocks_dict = get_text_fn("dict") if callable(get_text_fn) else {"blocks": []}
                if not (isinstance(blocks_dict, dict) and "blocks" in blocks_dict and isinstance(blocks_dict["blocks"], list)):
                    blocks = []
                else:
                    blocks = blocks_dict["blocks"]
                for blk in blocks:
                    if blk.get("type") != 0:
                        continue
                    for ln in blk.get("lines", []):
                        for sp in ln.get("spans", []):
                            span_box = fitz.Rect(sp["bbox"])
                            if span_box.intersects(bbox):
                                text += sp.get("text", "")
                links.append({
                    "page": page_num,
                    "bbox": bbox,
                    "uri": link["uri"],
                    "text": text.strip()
                })
    doc.close()
    return links

# ------------------- CONFIGURATION -------------------
INPUT_PDF = "test_manual.pdf"
OUTPUT_DIR = os.path.join("outputs", "test_manual")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "translated_test_manual.pdf")
JSON_FILE = os.path.join(OUTPUT_DIR, "translated_test_manual.json")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "..", "NotoSansJP-Regular.ttf")
EXTRACTED_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
DIST_THRESHOLD = 20
PADDING = 6
FONT_SIZE = 8
LOG_LEVEL = logging.INFO

if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont('NotoJP', FONT_PATH))
    reportlab_font_name = 'NotoJP'
else:
    reportlab_font_name = 'Helvetica'

pdfmetrics.registerFont(TTFont('NotoJP-Bold', 'NotoSansJP-Bold.ttf'))

logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def ensure_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def extract_images_with_spire(input_pdf_path: str, output_folder: str) -> List[Dict[str, Any]]:
    ensure_output_dir(output_folder)
    doc = PdfDocument()  # type: ignore
    try:
        doc.LoadFromFile(input_pdf_path)
        helper = PdfImageHelper()
        extracted_images = []
        idx = 0
        for i in range(doc.Pages.Count):
            page = doc.Pages.get_Item(i)
            infos = helper.GetImagesInfo(page)
            for info in infos:
                x, y = float(info.Bounds.X), float(info.Bounds.Y)
                w, h = float(info.Bounds.Width), float(info.Bounds.Height)
                fname = os.path.join(output_folder, f"page_{i+1}_img_{idx}.png")
                if os.path.exists(fname):
                    extracted_images.append({"page": i+1, "file": fname, "bbox": [x, y, x+w, y+h]})
                    idx += 1
                    continue
                img = info.Image
                try:
                    img.Save(fname)
                except Exception as e:
                    logger.warning(f"Failed to save image {fname}: {e}")
                    continue
                extracted_images.append({"page": i+1, "file": fname, "bbox": [x, y, x+w, y+h]})
                idx += 1
        return extracted_images
    except Exception as e:
        logger.error(f"Failed to load PDF: {e}")
        return []
    finally:
        try:
            doc.Dispose()
        except Exception:
            pass

def camelot_bbox_to_fitz(ct, page_height: float) -> fitz.Rect:
    x1, y1_top, x2, y2_bot = map(float, ct._bbox)
    y0 = page_height - y1_top
    y1 = page_height - y2_bot
    return fitz.Rect(x1, y1, x2, y0)

def extract_tables(input_pdf_path: str, translator: Any, font: fitz.Font, target_lang: str) -> List[Dict[str, Any]]:
    tables = []
    try:
        camelot_tables_raw = read_pdf(input_pdf_path, pages="all", flavor="lattice")
    except Exception as e:
        logger.error(f"Camelot failed: {e}")
        return tables
    doc_tmp = fitz.open(input_pdf_path)
    try:
        for ct in camelot_tables_raw:
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            if all(all(not str(cell).strip() for cell in row) for row in ct.df.values.tolist()):
                continue
            page_index = (ct.page or 1) - 1
            try:
                page_height = doc_tmp[page_index].rect.height
            except Exception as e:
                logger.warning(f"Failed to get page height for table: {e}")
                continue
            rect = camelot_bbox_to_fitz(ct, page_height)
            orig = ct.df.values.tolist()
            def safe_translate(cell):
                try:
                    result = translator.translate(str(cell) if cell is not None else "")
                    return result if isinstance(result, str) else str(cell) if cell is not None else ""
                except Exception as e:
                    logger.warning(f"Translation failed for cell: {e}")
                    return str(cell) if cell is not None else ""
            trans = [[safe_translate(cell) for cell in row] for row in orig]
            min_rows, min_cols = 2, 2
            non_empty_cells = sum(1 for row in orig for cell in row if str(cell).strip())
            total_cells = len(orig) * len(orig[0]) if orig else 0
            if (
                len(orig) < min_rows or
                len(orig[0]) < min_cols or
                (total_cells > 0 and non_empty_cells / total_cells < 0.5)
            ):
                continue
            tables.append({"page": ct.page, "rect": rect, "orig": orig, "trans": trans})
    finally:
        doc_tmp.close()
        del doc_tmp
        gc.collect()
    return tables

def get_font(font_path: str) -> fitz.Font:
    try:
        return fitz.Font(fontfile=font_path)
    except Exception as e:
        logger.warning(f"Failed to load font '{font_path}': {e}. Using default font.")
        return fitz.Font()

def render_table_as_image_reportlab(table_data, bbox, image_path, font_name='NotoJP', header_font_size=20, body_font_size=16, min_body_font_size=8):
    max_cols = max(len(row) for row in table_data)
    table_data_padded = [list(row) + [''] * (max_cols - len(row)) for row in table_data]
    col_count = max_cols
    n_rows = len(table_data_padded)
    total_width = bbox[2] - bbox[0] if bbox else 1000
    col_widths = [total_width / col_count] * col_count
    styles = getSampleStyleSheet()
    para_style = styles["BodyText"].clone('tablecell')
    para_style.fontName = font_name
    para_style.fontSize = body_font_size
    para_style.wordWrap = 'CJK'
    para_style.leading = body_font_size * 1.2
    para_style.spaceAfter = 0
    para_style.spaceBefore = 0
    para_style.alignment = 0
    table_data_wrapped = []
    for i, row in enumerate(table_data_padded):
        non_empty = [cell for cell in row if str(cell).strip()]
        if len(non_empty) == 1:
            table_data_wrapped.append([str(cell) if cell is not None else '' for cell in row])
        else:
            wrapped_row = []
            for cell in row:
                text = str(cell) if cell is not None else ''
                wrapped_row.append(Paragraph(text.replace('\n', '<br/>'), para_style))
            table_data_wrapped.append(wrapped_row)
    def get_max_text_width(text, font_size):
        from reportlab.pdfbase.pdfmetrics import stringWidth
        return stringWidth(text, font_name, font_size)
    row_font_sizes = [body_font_size] * n_rows
    for i, row in enumerate(table_data_padded):
        non_empty = [cell for cell in row if cell.strip()]
        if len(non_empty) == 1:
            row_font_sizes[i] = header_font_size
        else:
            min_size = body_font_size
            for j, cell in enumerate(row):
                text = str(cell)
                font_size = body_font_size
                while font_size > min_body_font_size and get_max_text_width(text, font_size) > col_widths[j] - 8:
                    font_size -= 1
                if font_size < min_size:
                    min_size = font_size
            row_font_sizes[i] = min_size
    t = Table(table_data_wrapped, colWidths=col_widths)
    style = TableStyle([])
    i = 0
    while i < n_rows:
        non_empty = [cell for cell in table_data_padded[i] if cell.strip()]
        if len(non_empty) == 1:
            style.add('SPAN', (0,i), (col_count-1,i))
            style.add('BACKGROUND', (0,i), (col_count-1,i), colors.HexColor('#cc0000'))
            style.add('TEXTCOLOR', (0,i), (col_count-1,i), colors.white)
            style.add('ALIGN', (0,i), (col_count-1,i), 'CENTER')
            style.add('VALIGN', (0,i), (col_count-1,i), 'MIDDLE')
            style.add('FONTNAME', (0,i), (col_count-1,i), 'NotoJP-Bold')
            style.add('FONTSIZE', (0,i), (col_count-1,i), header_font_size)
            style.add('FONTWEIGHT', (0,i), (col_count-1,i), 'bold')
            if i+1 < n_rows and len([cell for cell in table_data_padded[i+1] if cell.strip()]) > 1:
                style.add('ALIGN', (0,i+1), (col_count-1,i+1), 'CENTER')
                style.add('VALIGN', (0,i+1), (col_count-1,i+1), 'MIDDLE')
                style.add('FONTNAME', (0,i+1), (col_count-1,i+1), 'NotoJP-Bold')
                style.add('FONTSIZE', (0,i+1), (col_count-1,i+1), max(row_font_sizes[i+1], min_body_font_size))
                style.add('FONTWEIGHT', (0,i+1), (col_count-1,i+1), 'bold')
            i += 2
        else:
            style.add('ALIGN', (0,i), (col_count-1,i), 'LEFT')
            style.add('VALIGN', (0,i), (col_count-1,i), 'MIDDLE')
            style.add('FONTNAME', (0,i), (col_count-1,i), font_name)
            style.add('FONTSIZE', (0,i), (col_count-1,i), max(row_font_sizes[i], min_body_font_size))
            i += 1
    style.add('INNERGRID', (0,0), (-1,-1), 1, colors.black)
    style.add('LEFTPADDING', (0,0), (0,-1), 36)
    style.add('LEFTPADDING', (1,0), (-1,-1), 12)
    style.add('RIGHTPADDING', (0,0), (-1,-1), 12)
    style.add('TOPPADDING', (0,0), (-1,-1), 10)
    style.add('BOTTOMPADDING', (0,0), (-1,-1), 10)
    style.add('LEADING', (0,0), (-1,-1), 1.5*body_font_size)
    style.add('ROWHEIGHTS', (0,0), (-1,-1), 2.2*body_font_size)
    t.setStyle(style)
    very_large_height = 20000
    doc = SimpleDocTemplate('temp_table.pdf', pagesize=(total_width, very_large_height))
    elements = [t]
    doc.build(elements)  # type: ignore
    pdf_table = fitz.open('temp_table.pdf')
    pdf_page = pdf_table[0]
    mat = fitz.Matrix(2, 2)
    pix = pdf_page.get_pixmap(matrix=mat, alpha=False)  # type: ignore
    threshold = 245
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    mask = np.any(img < threshold, axis=2)
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if rows.size and cols.size:
        y0, y1 = rows[0], rows[-1]
        x0, x1 = cols[0], cols[-1]
        margin = 0
        y0 = max(0, y0 - margin)
        y1 = min(img.shape[0] - 1, y1 + margin)
        x0 = max(0, x0 - margin)
        x1 = min(img.shape[1] - 1, x1 + margin)
        cropped = img[y0:y1+1, x0:x1+1]
        im = Image.fromarray(cropped)
        border_width = 1
        border_color = (0, 0, 0)
        from PIL import ImageOps
        im = ImageOps.expand(im, border=border_width, fill=border_color)
        im.save(image_path)
    else:
        pix.save(image_path)
    pdf_table.close()
    os.remove('temp_table.pdf')

def render_tables(npg, page_tables, font, pad, fsize, images_dir, out_pdf=None, page_rect=None, page_num=None):
    for t_idx, t in enumerate(page_tables):
        rect = t["rect"]
        bbox = [0, 0, rect.width * 2, rect.height]
        image_path = os.path.join(images_dir, f"table_img_page{page_num}_table{t_idx}_{uuid4().hex[:8]}.png")
        render_table_as_image_reportlab(
            t["trans"],
            bbox,
            image_path,
            font_name=reportlab_font_name,
            header_font_size=24,
            body_font_size=18
        )
        fit_rect = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y1)
        npg.insert_image(fit_rect, filename=image_path)

def process_pdf(input_pdf: str, output_pdf: str, json_file: str, font_path: str, translator: Any, dist_thr: int, pad: int, fsize: int, target_lang: str = "ja", max_pages: Optional[int] = None, should_stop=None) -> None:
    os.add_dll_directory(os.getcwd())
    font = get_font(font_path)
    output_dir = os.path.dirname(output_pdf)
    images_dir = os.path.join(output_dir, "images")
    images = extract_images_with_spire(input_pdf, images_dir)
    tables = extract_tables(input_pdf, translator, font, target_lang)
    out_pdf = fitz.open()
    doc_in = fitz.open(input_pdf)
    metadata = []
    translator_api_calls = 0
    original_rects = extract_rects(input_pdf)
    hyperlinks = extract_hyperlinks(input_pdf)
    num_pages = len(doc_in)
    if max_pages is not None:
        num_pages = min(num_pages, max_pages)
    for pnum in range(num_pages):
        spans = []
        if should_stop and should_stop():
            logger.info("Stopping PDF processing early due to user request.")
            break
        page = doc_in.load_page(pnum)
        try:
            npg = out_pdf.new_page(width=page.rect.width, height=page.rect.height)  # type: ignore
        except AttributeError:
            npg = out_pdf.newPage(width=page.rect.width, height=page.rect.height)  # type: ignore
        npg.insert_font(fontname="NotoJP", fontfile=font_path)
        npg.insert_font(fontname="NotoJP-Bold", fontfile="NotoSansJP-Bold.ttf")
        page_data = {"page": pnum+1, "images": [], "labels": [], "spans": []}
        page_tables = [t for t in tables if t["page"] == pnum+1]
        if page_tables:
            page_data["tables"] = []
            render_tables(npg, page_tables, font, pad, fsize, images_dir, out_pdf=out_pdf, page_rect=page.rect, page_num=pnum+1)
            for t in page_tables:
                rect = t["rect"]
                page_data["tables"].append({
                    "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                    "orig_data": t["orig"],
                    "trans_data": t["trans"]
                })
        table_rects = [t["rect"] for t in page_tables]
        text_bboxes = [s['bbox'] for s in spans]
        table_bboxes = [t['rect'] for t in tables if t['page'] == pnum+1]
        all_bboxes = text_bboxes + [[tb.x0, tb.y0, tb.x1, tb.y1] for tb in table_bboxes]
        margin = 10
        page_images = [img for img in images if img['page'] == pnum+1]
        page_images.sort(key=lambda img: img['bbox'][1])
        for img in images:
            if img["page"] == pnum+1:
                x0, y0, x1, y1 = img["bbox"]
                try:
                    npg.insert_image(fitz.Rect(x0, y0, x1, y1), filename=img["file"])
                    page_data["images"].append({"file": img["file"], "bbox": img["bbox"]})
                except Exception as e:
                    logger.warning(f"Failed to insert image {img['file']}: {e}")
        try:
            blocks = page.get_text("dict")["blocks"]  # type: ignore
        except AttributeError:
            blocks = page.getText("dict") ["blocks"]  # type: ignore
        span_entries = []
        for blk in blocks:
            if blk.get("type") != 0:
                continue
            for ln in blk.get("lines", []):
                for sp in ln.get("spans", []):
                    if should_stop and should_stop():
                        logger.info("Stopping PDF processing early during span translation due to user request.")
                        return
                    raw = sp.get("text", "").strip()
                    if not raw:
                        continue
                    span_box = fitz.Rect(sp["bbox"])
                    if any(span_box.intersects(trect) for trect in table_rects):
                        continue
                    span_entries.append({
                        "orig": raw,
                        "bbox": sp["bbox"],
                        "size": sp.get("size"),
                        "color": sp.get("color"),
                        "font": sp.get("font"),
                        "flags": sp.get("flags"),
                    })
        if hasattr(translator, 'batch_translate'):
            texts = [entry["orig"] for entry in span_entries]
            translations = translator.batch_translate(texts)
            for entry, trans in zip(span_entries, translations):
                entry["trans"] = trans
        else:
            for entry in span_entries:
                entry["trans"] = translator.translate(entry["orig"])
        spans.extend(span_entries)
        clusters, used = [], set()
        for i, s1 in enumerate(spans):
            if i in used: continue
            grp = [s1]; used.add(i)
            r1 = fitz.Rect(s1["bbox"])
            infl = fitz.Rect(r1.x0-dist_thr, r1.y0-dist_thr, r1.x1+dist_thr, r1.y1+dist_thr)
            for j, s2 in enumerate(spans):
                if j not in used and infl.intersects(fitz.Rect(s2["bbox"])):
                    grp.append(s2); used.add(j)
            clusters.append(grp)
        page_data["labels"] = []
        rects = [fitz.Rect(s["bbox"]) for s in spans]
        merged, used_idx = [], set()
        for i, r in enumerate(rects):
            if i in used_idx: continue
            grp = [r]; used_idx.add(i); changed=True
            while changed:
                changed=False
                for j, s in enumerate(rects):
                    if j not in used_idx and any(g.intersects(s) for g in grp):
                        grp.append(s); used_idx.add(j); changed=True
            x0 = min(g.x0 for g in grp); y0 = min(g.y0 for g in grp)
            x1 = max(g.x1 for g in grp); y1 = max(g.y1 for g in grp)
            merged.append(fitz.Rect(x0, y0, x1, y1))
        page_data["labels"] = [{
            "bbox":[r.x0,r.y0,r.x1,r.y1],
            "orig_text":"\n".join(str(s.get("orig") or "") for grp in clusters for s in grp if fitz.Rect(s["bbox"]).intersects(r)),
            "translated_text":"\n".join(str(s.get("trans") or s.get("orig") or "") for grp in clusters for s in grp if fitz.Rect(s["bbox"]).intersects(r))
        } for r in merged]
        page_data["spans"] = [
            {"orig": s["orig"], "trans": s["trans"], "bbox": s["bbox"]} for s in spans
        ]
        def int_to_rgb(color_int):
            r = (color_int >> 16) & 255
            g = (color_int >> 8) & 255
            b = color_int & 255
            return (r/255, g/255, b/255)
        image_bboxes = [fitz.Rect(*img["bbox"]) for img in images if img["page"] == pnum+1]
        table_bboxes = [t["rect"] for t in tables if t["page"] == pnum+1]
        pipeline_boxes_drawn = []
        rendered_span_indices = set()
        pipeline_boxes = []
        for rect in original_rects:
            if rect["page"] != pnum:
                continue
            pipeline_box = fitz.Rect(rect["x0"], rect["y0"], rect["x1"], rect["y1"])
            if any(pipeline_box.intersects(table_box) for table_box in table_bboxes):
                continue
            merged_spans = []
            for idx, s in enumerate(spans):
                span_box = fitz.Rect(s["bbox"])
                if span_box.intersects(pipeline_box):
                    merged_spans.append((idx, s))
            if merged_spans:
                merged_spans.sort(key=lambda tup: (tup[1]["bbox"][1], tup[1]["bbox"][0]))
                merged_text = ""
                for i, (idx, s) in enumerate(merged_spans):
                    text = s.get("orig") or ""
                    if i > 0:
                        prev = merged_spans[i-1][1]
                        prev_bottom = prev["bbox"][3]
                        curr_top = s["bbox"][1]
                        if curr_top - prev_bottom > fsize * 0.8:
                            merged_text += "\n"
                        else:
                            merged_text += " "
                    merged_text += text
                pipeline_boxes.append({
                    "rect": pipeline_box,
                    "span_indices": [idx for idx, _ in merged_spans],
                    "merged_text": merged_text
                })
        merged_texts = [box["merged_text"] for box in pipeline_boxes]
        if hasattr(translator, 'batch_translate'):
            translated_texts = translator.batch_translate(merged_texts)
            translator_api_calls += 1
        else:
            translated_texts = []
            for text in merged_texts:
                translated_texts.append(translator.translate(text))
                translator_api_calls += 1
        for box, trans in zip(pipeline_boxes, translated_texts):
            box["translated_text"] = trans
        for box in pipeline_boxes:
            for idx in box["span_indices"]:
                rendered_span_indices.add(idx)
        for box in pipeline_boxes:
            pipeline_box = box["rect"]
            merged_text = box["translated_text"]
            def is_cjk(text):
                for ch in text:
                    if '\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff':
                        return True
                return False
            lines = []
            if is_cjk(merged_text):
                for para in merged_text.split("\n"):
                    line = ""
                    for ch in para:
                        test = line + ch
                        if font.text_length(test, fontsize=fsize) <= pipeline_box.width - 2*pad:
                            line = test
                        else:
                            lines.append(line)
                            line = ch
                    if line:
                        lines.append(line)
            else:
                for para in merged_text.split("\n"):
                    words = para.split()
                    line = ""
                    for word in words:
                        test = (line + " " + word).strip() if line else word
                        if font.text_length(test, fontsize=fsize) <= pipeline_box.width - 2*pad:
                            line = test
                        else:
                            if line:
                                lines.append(line)
                            line = word
                    if line:
                        lines.append(line)
            seen_lines = set()
            deduped_lines = []
            for ln in lines:
                if ln not in seen_lines:
                    deduped_lines.append(ln)
                    seen_lines.add(ln)
            lines = deduped_lines
            line_height = fsize * 1.7
            total_height = len(lines) * line_height
            box_rect = pipeline_box
            if total_height > box_rect.height:
                box_rect = fitz.Rect(box_rect.x0, box_rect.y0, box_rect.x1, box_rect.y0 + total_height)
            y0 = box_rect.y0 + pad * 2
            color = (0,0,0)
            for ln in lines:
                text_width = font.text_length(ln, fontsize=fsize)
                x_center = box_rect.x0 + (box_rect.width - text_width) / 2
                npg.insert_text((x_center, y0), ln, fontname="NotoJP", fontsize=fsize, color=color)
                y0 += line_height
            npg.draw_rect(box_rect, color=(1,0,0), width=1)
            pipeline_boxes_drawn.append(pipeline_box)
        image_margin = 0
        for rect in original_rects:
            if rect["page"] != pnum:
                continue
            rect_box = fitz.Rect(rect["x0"], rect["y0"], rect["x1"], rect["y1"])
            for img_box in image_bboxes:
                shrunk_img_box = fitz.Rect(
                    img_box.x0 + image_margin,
                    img_box.y0 + image_margin,
                    img_box.x1 - image_margin,
                    img_box.y1 - image_margin
                )
                if rect_fully_contained(rect_box, shrunk_img_box):
                    npg.draw_rect(rect_box, color=(1,0,0), width=1)
                    break
        non_box_spans = [s for idx, s in enumerate(spans) if idx not in rendered_span_indices]
        if hasattr(translator, 'batch_translate') and non_box_spans:
            texts = [s["orig"] for s in non_box_spans]
            translations = translator.batch_translate(texts)
            translator_api_calls += 1
            for s, trans in zip(non_box_spans, translations):
                s["trans"] = trans
        else:
            for s in non_box_spans:
                s["trans"] = translator.translate(s["orig"])
                translator_api_calls += 1
        for s in non_box_spans:
            orig_text = s.get("trans") or s.get("orig") or ""
            bbox = s["bbox"]
            box = fitz.Rect(*bbox)
            def is_cjk(text):
                for ch in text:
                    if '\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff':
                        return True
                return False
            lines = []
            if is_cjk(orig_text):
                line = ""
                for ch in orig_text:
                    test = line + ch
                    if font.text_length(test, fontsize=fsize) <= box.width - 2*pad:
                        line = test
                    else:
                        lines.append(line)
                        line = ch
                if line:
                    lines.append(line)
            else:
                words = orig_text.split()
                line = ""
                for word in words:
                    test = (line + " " + word).strip() if line else word
                    if font.text_length(test, fontsize=fsize) <= box.width - 2*pad:
                        line = test
                    else:
                        if line:
                            lines.append(line)
                        line = word
                if line:
                    lines.append(line)
            seen_lines = set()
            deduped_lines = []
            for ln in lines:
                if ln not in seen_lines:
                    deduped_lines.append(ln)
                    seen_lines.add(ln)
            lines = deduped_lines
            line_height = fsize * 1.7
            total_height = len(lines) * line_height
            if total_height > box.height:
                box = fitz.Rect(box.x0, box.y0, box.x1, box.y0 + total_height)
            y0 = box.y0 + pad * 2
            color = (0,0,0)
            if s.get("color") is not None:
                color_int = s["color"]
                color = (
                    ((color_int >> 16) & 255) / 255,
                    ((color_int >> 8) & 255) / 255,
                    (color_int & 255) / 255
                )
            flags = s.get("flags", 0)
            fontname = "NotoJP-Bold" if (flags & 8) else "NotoJP"
            for ln in lines:
                text_width = font.text_length(ln, fontsize=fsize)
                x_center = box.x0 + (box.width - text_width) / 2
                npg.insert_text((x_center, y0), ln, fontname=fontname, fontsize=fsize, color=color)
                y0 += line_height
        metadata.append(page_data)
    out_pdf.save(output_pdf)
    out_pdf.close()
    doc_in.close()
    del doc_in
    del out_pdf
    gc.collect()
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(metadata, jf, ensure_ascii=False, indent=2)
    logger.info(f"Translation and aligned boxes (including tables) complete: {output_pdf}")

def main(max_pages: Optional[int]):
    logger.info("Using Madlad400_3B_MT_Translator.")
    ensure_output_dir(OUTPUT_DIR)
    ensure_output_dir(EXTRACTED_IMG_DIR)
    TRANSLATION_CACHE_PATH = os.path.join(OUTPUT_DIR, "translation_cache.json")
    warnings.filterwarnings("ignore", message=".*does not lie in column range.*")
    logger.info("Starting PDF translation pipeline...")
    process_pdf(
        input_pdf=INPUT_PDF,
        output_pdf=OUTPUT_PDF,
        json_file=JSON_FILE,
        font_path=FONT_PATH,
        translator=None,
        dist_thr=DIST_THRESHOLD,
        pad=PADDING,
        fsize=FONT_SIZE,
        max_pages=max_pages
    )

if __name__ == "__main__":
    raise NotImplementedError("Please provide an API key when calling main().") 