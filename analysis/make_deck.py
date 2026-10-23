"""Builds the supervisor-meeting deck from figures/ (run analysis.slide_figures first).

    python -m analysis.make_deck        # writes MTP_progress.pptx next to the repo
"""
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "..", "MTP_progress.pptx")

INK = RGBColor(0x22, 0x2B, 0x33)
MUTED = RGBColor(0x5F, 0x6B, 0x75)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
BOX = RGBColor(0xEE, 0xF2, 0xF6)
LINE = RGBColor(0x8A, 0x96, 0xA3)
FONT = "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W = 13.333


def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """runs: a string, or a list of paragraphs; a paragraph is a string or a list of (text, overrides) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(6)
        for t, o in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = t
            f = r.font
            f.name, f.size = FONT, Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.color.rgb = o.get("color", color)
    return tb


def new_slide(title, notes=None):
    s = prs.slides.add_slide(BLANK)
    text(s, 0.6, 0.4, W - 1.2, 0.8, title, size=32, bold=True)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    text(s, W - 1.1, 7.0, 0.6, 0.3, str(len(prs.slides)), size=10, color=MUTED, align=PP_ALIGN.RIGHT)
    return s


def box(slide, x, y, w, h, label, size=16, fill=BOX, color=INK, bold=False):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.12
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = LINE
    shp.line.width = Pt(1)
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    lines = label.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = ln
        r.font.name, r.font.size, r.font.color.rgb = FONT, Pt(size if i == 0 else size - 3), color
        r.font.bold = bold if i == 0 else False
    return shp


def arrow(slide, x1, y1, x2, y2):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = LINE
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle"})
    ln.append(tail)
    return c


def picture(slide, name, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(os.path.join(FIG, name), Inches(x), Inches(y), **kw)


def stat(slide, x, y, w, number, label):
    text(slide, x, y, w, 0.7, number, size=30, bold=True, color=ACCENT)
    text(slide, x, y + 0.65, w, 0.7, label, size=14, color=MUTED)


def table(slide, x, y, w, rows, col_w, size=15, row_h=0.45):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    t = shp.table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            c.fill.solid()
            c.fill.fore_color.rgb = ACCENT if i == 0 else (BOX if i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF))
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = val
            r.font.name, r.font.size = FONT, Pt(size)
            r.font.bold = i == 0
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else INK
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
    return t


# 1 Title
s = prs.slides.add_slide(BLANK)
text(s, 0.9, 1.8, 11.8, 2.2, ["Coordinated Control of Speculative Decoding,",
                              "KV-Cache Management, and Admission",
                              "for Robust LLM Serving"], size=36, bold=True)
text(s, 0.9, 4.2, 11.5, 1.6, ["Mohsin Sheikh Mubarak (M25AI2074)",
                              "Supervisor: Prof. Navchetan Awasthi",
                              "M.Tech project, School of AI and Data Science, IIT Jodhpur  ·  progress review, 24 Oct 2026"],
     size=18, color=MUTED)

# 2 Problem
s = new_slide("Three knobs, one memory budget",
              "Production systems set gamma, KV policy and admission separately and statically. "
              "The 1B drafter's weights and KV take memory the target KV cache could use, so gamma also changes "
              "how many requests fit.")
for i, (lab, sub) in enumerate([("Speculative decoding", "how many tokens to draft (γ)"),
                                ("KV cache", "what stays on GPU, what is evicted"),
                                ("Admission / scheduling", "who runs, batch size, preemption")]):
    bx = 0.9 + i * 4.0
    box(s, bx, 1.9, 3.5, 1.3, f"{lab}\n{sub}", size=18, bold=True)
    arrow(s, bx + 1.75, 3.2, bx + 1.75, 4.2)
box(s, 0.9, 4.2, 11.5, 1.0, "GPU memory  (22 GB budget: target weights + draft model + KV cache)", size=18, bold=True)
text(s, 0.9, 5.7, 11.5, 1.0, "Usually tuned separately and fixed. But a draft model costs KV capacity, "
     "so choosing γ is also choosing how many requests fit.", size=18, color=MUTED)

# 3 Evidence
s = new_slide("No fixed setting is right everywhere",
              "Stock vLLM 0.31 on A100, 22 GB. Saturation: no-spec 13.37 (3 reps); gamma 1/2/4/6/8 = 3.28/5.16/2.82/2.15/0.04 "
              "(single runs). At 1 req/s, gamma=2 gives 6.8 ms vs 10.9 ms per token.")
picture(s, "1_static_spec_regimes.png", 0.6, 1.35, w=8.6)
stat(s, 9.7, 1.8, 3.2, "13.4 → 0.04–5.2", "goodput under load, no speculation vs any fixed γ")
stat(s, 9.7, 3.6, 3.2, "10.9 → 6.8 ms", "per-token latency at 1 req/s, no speculation vs γ=2")

# 4 What I built
s = new_slide("What I built",
              "Engine written from scratch, uses vLLM's fused kernels (FlashAttention, RMSNorm, RoPE, SiLU, KV write). "
              "Snapshot every 8 steps; one pressure level shared by all controllers. Parking hands the drafter's "
              "memory to the KV pool; unparking migrates live blocks and restores weights from pinned host memory.")
box(s, 0.6, 1.8, 2.6, 1.2, "Engine step\nbatch, draft, verify", size=17, bold=True)
box(s, 3.8, 1.8, 2.6, 1.2, "Shared snapshot\nevery 8 steps", size=17, bold=True)
box(s, 7.0, 1.8, 2.6, 1.2, "Co-optimizer\none pressure level", size=17, bold=True, fill=ACCENT, color=RGBColor(0xFF, 0xFF, 0xFF))
arrow(s, 3.2, 2.4, 3.8, 2.4)
arrow(s, 6.4, 2.4, 7.0, 2.4)
for i, lab in enumerate(["γ controller\nγ per step, learned", "KV controller\npark / restore drafter",
                         "Scheduler controller\nbatch cap, admission"]):
    by = 1.4 + i * 1.05
    box(s, 10.2, by, 2.6, 0.9, lab, size=15, bold=True)
    arrow(s, 9.6, 2.4, 10.2, by + 0.45)
text(s, 0.6, 4.6, 6.0, 2.2, ["Built from scratch, runs vLLM's own GPU kernels",
                             "So any difference comes from control, not kernels",
                             "Paged KV cache with prefix caching, CUDA graphs, OpenAI-compatible server"],
     size=16)
text(s, 7.0, 4.6, 5.8, 2.2, [[("Drafter parking: ", {"bold": True}),
                              ("under pressure, the draft model's memory is given to the KV cache, "
                               "and taken back when load drops", {})]], size=16)

# 5 Correctness
s = new_slide("The engine is correct",
              "M4 correctness on A100: teacher-forced agreement 99.16 %, same as HF eager vs SDPA. "
              "Rejection sampling within 1 % over 40k samples. Elastic check PASS on a clean GPU (first try OOM'd "
              "because another process held 23 GB). INT8 6.7982 vs bf16 6.7941 perplexity (4090, GPU-independent).")
table(s, 0.9, 1.7, 11.5, [
    ["Check", "Result"],
    ["Next-token agreement with HuggingFace (Llama-3.1-8B)", "99.16 %  (HF vs itself across kernels: 99.16 %)"],
    ["Speculative decoding output", "Lossless (rejection-sampling test)"],
    ["Drafter park / unpark during generation (A100)", "0 of 6,144 tokens confidently wrong"],
    ["INT8 KV cache", "+0.06 % perplexity"],
    ["Unit tests", "16 / 16 pass"],
], [6.5, 5.0], size=16, row_h=0.6)

# 6 Setup
s = new_slide("Setup",
              "vLLM 0.31.0, PyTorch 2.13, CUDA 13.0. Temperature 0, ignore_eos so every system generates the same "
              "lengths. SLO: TTFT <= 1 s short / 10 s long, TPOT <= 100 ms.")
table(s, 0.9, 1.7, 11.5, [
    ["", "Details"],
    ["Hardware", "A100 80 GB, every system capped at 22 GB"],
    ["Models", "Llama-3.1-8B target, Llama-3.2-1B draft (BF16)"],
    ["Baseline", "vLLM 0.31, same kernels, same batch defaults, same drafter"],
    ["Workloads", "chat at 1, 4, 16 req/s and saturation · long-context · mixed-bursty · phase-shift"],
    ["Metric", "goodput = requests/s meeting the latency SLO"],
    ["Repeats", "3 per configuration (ablations: 1)"],
], [2.5, 9.0], size=16, row_h=0.6)

# 7 Coordination
s = new_slide("Coordination pays",
              "Same three controllers; only the shared view differs. Original: 16 rps 8.46 -> 9.63, saturation 7.74 -> 10.67, "
              "phase-shift 2.32 -> 2.96. With fix: 9.52 -> 9.96, phase-shift 2.35 -> 3.15, burst SLO 0.07 -> 0.42. "
              "'Indep. + fix' at saturation (13.04) is inflated: 2 of 3 repeats started with the drafter already parked "
              "from the previous run.")
picture(s, "2_coordination.png", 0.4, 1.35, w=12.5)
stat(s, 0.9, 5.0, 3.6, "+34 %", "phase-shift goodput, with the fix (3.15 vs 2.35)")
stat(s, 4.9, 5.0, 3.6, "0.42 vs 0.07", "share of burst requests meeting the SLO")
stat(s, 8.9, 5.0, 3.8, "+38 %", "chat saturation, original version (clean comparison)")

# 8 Trace
s = new_slide("The controller adapting, phase by phase",
              "Chat: gamma about 3-4. Long documents: parks and unparks as memory tightens (flips a few times). "
              "Burst: parks at the start; queue peaks around 135 and drains. Recovery: speculation back about 8 s "
              "after the queue empties.")
picture(s, "5_phase_shift_trace.png", 0.5, 1.3, h=5.7)
text(s, 10.2, 1.6, 2.7, 5.0, [[("Chat", {"bold": True})], "speculates, γ ≈ 3–4",
                              [("Long documents", {"bold": True})], "parks when memory is tight",
                              [("Burst", {"bold": True})], "parks, queue drains",
                              [("Recovery", {"bold": True})], "speculation resumes"], size=16)

# 9 Burst fix
s = new_slide("Burst fix: back off on preemptions only",
              "Old rule backed off the batch cap whenever GPU memory was >= 97 % full, which is normal during a burst, "
              "and grew back only when calm, so the cap sat near 8. New rule: back off on preemptions only. "
              "All numbers co-optimizer, 3 repeats.")
text(s, 0.9, 1.5, 11.5, 0.9, "Before: the batch cap shrank whenever GPU memory was ≥ 97 % full, which is normal in a burst. "
     "Now it shrinks only when requests are preempted.", size=17, color=MUTED)
table(s, 0.9, 2.6, 8.0, [
    ["Co-optimizer, goodput", "Before", "After"],
    ["Mixed-bursty", "0.53", "0.67"],
    ["Phase-shift", "2.96", "3.15"],
    ["Burst-phase SLO", "0.34", "0.42"],
    ["Chat 16 req/s", "9.63", "9.96"],
    ["Chat saturation", "10.67", "9.72"],
], [4.0, 2.0, 2.0], size=17, row_h=0.6)
stat(s, 9.6, 2.8, 3.3, "+26 %", "mixed-bursty goodput (vLLM: 0.72)")
stat(s, 9.6, 4.5, 3.3, "−9 %", "chat saturation: the trade-off")

# 10 Ablation
s = new_slide("Parking enables it, coordination makes it pay",
              "Chat saturation goodput. Ablations are single runs; co-optimizer and independent are 3 repeats.")
picture(s, "3_ablation.png", 0.5, 1.5, w=8.9)
text(s, 9.8, 1.8, 3.1, 4.5, [[("Without parking", {"bold": True})], "2.7–4.1 goodput",
                             [("KV controller + parking", {"bold": True})], "4.3",
                             [("All three, coordinated", {"bold": True})], "10.7",
                             [("Single runs, except the top two", {"size": 13, "color": MUTED})]], size=17)

# 11 Baselines
s = new_slide("Against baselines",
              "All 3 repeats. Light load (not in figure): 1 req/s TPOT p50 6.9 ms co-opt + fix vs 12.5 ms own engine "
              "without speculation vs 10.9 ms vLLM. Our engine without speculation trails vLLM by 5-14 % under load.")
picture(s, "4_baselines.png", 0.4, 1.3, w=12.5)
stat(s, 0.9, 5.0, 3.8, "6.9 vs 12.5 ms", "per token at light load, vs the same engine without speculation")
stat(s, 4.9, 5.0, 3.8, "9.72 vs 2.38", "saturation goodput, vs vLLM's adaptive speculation")
stat(s, 8.9, 5.0, 3.8, "5–14 %", "how far our engine trails vLLM under load, with or without control")

# 12 Limitations
s = new_slide("Limitations",
              "Carry-over: the drafter's parked state persists from one run to the next on the same server; runs that "
              "start parked score 11-14 at saturation. Fix: reset state before each run, re-run saturation.")
for i, (head, body) in enumerate([
        ("Carry-over between runs", "Drafter state persists across runs on one server and inflates some saturation "
                                    "numbers. Fix: reset before each run."),
        ("Engine gap", "Our engine trails stock vLLM by 5–14 % under load."),
        ("Scope", "One model pair, one GPU, greedy decoding."),
        ("Nightjar-like baseline", "Our re-implementation of the idea, not the authors' system.")]):
    bx, by = 0.9 + (i % 2) * 5.9, 1.7 + (i // 2) * 2.5
    shp = box(s, bx, by, 5.5, 2.1, "", size=16)
    text(s, bx + 0.3, by + 0.25, 4.9, 1.7, [[(head, {"bold": True, "size": 19})], body], size=16)

# 13 Status and plan
s = new_slide("Status and next steps",
              "Saturation re-run with state reset: estimate about 1 GPU-hour. Ask: title and framing, which version "
              "is the system, deadline and format, EuroMLSys 2027 workshop / arXiv.")
text(s, 0.9, 1.6, 5.5, 0.5, "Done", size=22, bold=True, color=ACCENT)
text(s, 0.9, 2.2, 5.5, 4.0, ["Engine, three controllers, drafter parking",
                             "Validation: correctness, lossless speculation, quality",
                             "All A100 experiments, 3 repeats",
                             "Burst fix, measured"], size=17)
text(s, 7.0, 1.6, 5.5, 0.5, "Next", size=22, bold=True, color=ACCENT)
text(s, 7.0, 2.2, 5.5, 2.0, ["Re-run chat saturation with a state reset",
                             "Chapter 3: formal model",
                             "Thesis writing"], size=17)
text(s, 7.0, 4.3, 5.5, 0.5, "To discuss", size=22, bold=True, color=ACCENT)
text(s, 7.0, 4.9, 5.5, 2.2, ["Title and the coordination framing",
                             "Present the fixed version as the system?",
                             "Deadline and format; workshop paper or arXiv"], size=17)

prs.save(OUT)
print("wrote", os.path.abspath(OUT))
