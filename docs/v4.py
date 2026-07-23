import pathlib

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, PageBreak)

DARK = HexColor("#16161a")
ACCENT = HexColor("#264653")     # deep teal — a "night lighting" mood
ACCENT2 = HexColor("#e76f51")    # warm ember
MUT = HexColor("#555555")
LIGHT_BG = HexColor("#eef1f2")
RULE = HexColor("#b9c4c9")

styles = getSampleStyleSheet()

title_s = ParagraphStyle("TitleS", parent=styles["Title"], fontName="Helvetica-Bold",
                         fontSize=23, textColor=DARK, spaceAfter=4, alignment=TA_LEFT)
subtitle_s = ParagraphStyle("SubS", parent=styles["Normal"], fontName="Helvetica",
                            fontSize=11.5, textColor=MUT, spaceAfter=14, leading=15)
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                    fontSize=15, textColor=ACCENT, spaceBefore=16, spaceAfter=6)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                    fontSize=12.5, textColor=DARK, spaceBefore=10, spaceAfter=4)
body = ParagraphStyle("BodyS", parent=styles["Normal"], fontName="Helvetica",
                      fontSize=10.3, leading=14.5, textColor=DARK, spaceAfter=6)
bullet = ParagraphStyle("BulletS", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=4)
mile = ParagraphStyle("Mile", parent=body, leftIndent=14, spaceAfter=5)
goal = ParagraphStyle("Goal", parent=body, leftIndent=14, textColor=HexColor("#8a4a24"),
                      fontName="Helvetica-Oblique", spaceAfter=6)

# Written next to this script, so the PDF lands in docs/ however the script was invoked.
_OUT = pathlib.Path(__file__).with_name("AI_Civilization_Visual_Overhaul_v4.pdf")

doc = SimpleDocTemplate(str(_OUT),
                        pagesize=A4, topMargin=18*mm, bottomMargin=18*mm,
                        leftMargin=18*mm, rightMargin=18*mm,
                        title="AI Civilization — Visual Overhaul v4",
                        author="Prateek Raj")

S = []

def rule():
    t = Table([[""]], colWidths=[doc.width], rowHeights=[0.8])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.8, RULE)]))
    return t

def slice_item(sid, name, text):
    return Paragraph(f"<b>{sid} — {name}.</b> {text}", mile)

# ---------------- COVER ----------------
S.append(Paragraph("AI Civilization — Visual Overhaul v4", title_s))
S.append(Paragraph("From 'detailed but unreadable' to 'genuinely cool'. A showcase-grade visual plan: "
                   "legibility first, then a crafted palette, real lighting, an isometric world, and juice. "
                   "All procedural Pygame — no sprite art, no asset pipeline. ~1 month at 45–60 min/day.",
                   subtitle_s))
S.append(rule())
S.append(Spacer(1, 6))

S.append(Paragraph("The honest diagnosis", h2))
S.append(Paragraph(
    "The simulation is complete (v5.0, 294 tests, all six arcs + minds-at-the-pivots) and the renderer has "
    "plenty of detail — but the current look fails the showcase test for four reasons: <b>noise over signal</b> "
    "(the event feed is trust-delta spam; major events drown), <b>flat rendering</b> (top-down circles and icons "
    "read as a diagram, not a world), <b>undisciplined colour</b> (many saturated hues at similar values turn "
    "overlapping settlements into mush), and <b>weak lighting</b> (night is a dark tint, not darkness with light "
    "in it). The fix is not more objects — it is rendering craft: hierarchy, palette, light, projection, motion.",
    body))

S.append(Paragraph("Design north star", h2))
S.append(Paragraph(
    "A stranger watching only the map should follow the story beat by beat — and want to keep watching. "
    "Reference feel: the readable-clean of Mini Metro, the mood of a lantern-lit strategy map, the depth of a "
    "2.5D isometric world. Simple shapes, crafted like a designed game — not decorated like a tech demo.",
    body))

S.append(Paragraph("Ground rules (unchanged)", h2))
for b in [
    "Renderer stays READ-ONLY; byte-identical seeded sim runs; AST boundary green; all 294 tests pass every slice.",
    "All visuals procedural (shapes, polygons, gradients) — no image assets, no sprite art.",
    "One slice at a time, committed clean; visual slices are verified by EYES plus the suite, never the suite alone.",
    "Performance budget: smooth at grid 24 / 30 agents (~10ms frame); cache aggressively, per-frame cost stays flat.",
]:
    S.append(Paragraph(f"• {b}", bullet))

S.append(PageBreak())

# ---------------- WEEK 1 ----------------
S.append(Paragraph("Week 1 — Legibility &amp; Signal (fix what's broken first)", h1))
S.append(Paragraph("Iso-projection on an unreadable scene is lipstick on noise. Week 1 makes the current look "
                   "READ, and fixes two live bugs.", body))
S.append(slice_item("V4.1", "HUD + Realms bugfix",
    "The bottom HUD draws overlapping garbled text; the REALMS panel shows '(none)' while STATE shows kingdoms 2 / "
    "empires 1. Fix HUD layout (proper spacing at any zoom) and wire the realm scoreboard to actually read "
    "kingdoms/empires with settlement counts in realm colours."))
S.append(slice_item("V4.2", "Event tiers + the story banner",
    "MAJOR events (battles, coronations, uprisings, secessions, empire formed/fragmented, era advances, faith "
    "founded, ruler deaths) always show, larger and bolder; MINOR noise (per-agent trust deltas, routine trades) "
    "is aggregated to one line per turn or suppressed. A prominent top-of-map BANNER announces each major event "
    "in plain words ('KING Borin's host was REPELLED at S0A2') — the single feature that makes the map "
    "self-explanatory without the side panel."))
S.append(slice_item("V4.3", "Territory clarity",
    "Overlapping settlement circles currently blur into a blob. Distinct outlined regions: stronger boundary "
    "stroke, lower fill opacity, realm-coloured edges; labels collision-nudged off buildings. Night floor "
    "brightness raised and weather particle density/opacity reduced so atmosphere never obscures the story."))
S.append(Paragraph("End state: the current renderer, readable — a viewer follows the story from map + banner alone.", goal))

# ---------------- WEEK 2 ----------------
S.append(Paragraph("Week 2 — Palette &amp; Light (the crafted look)", h1))
S.append(slice_item("V4.4", "Palette discipline",
    "Pick 4–5 base hues and push every element toward them; desaturate the commons; reserve saturation for what "
    "matters (rulers, banners, battles, the banner strip). One PALETTE pass over terrain, buildings, agents, "
    "territory and UI so the whole frame reads as one designed image. This is taste work — expect 2–3 tuning "
    "evenings with screenshots side by side."))
S.append(slice_item("V4.5", "Real light sources",
    "Replace the flat night tint with LIGHT AS A SYSTEM: windows, torches, hearths and the castle brazier become "
    "point lights casting warm radial gradients onto the ground (cached gradient surfaces, additive blend). A town "
    "at night becomes a cluster of warm pools in blue darkness; dawn dissolves the pools. Battles at night are lit "
    "by the clash flashes. This is the single biggest MOOD upgrade available per hour spent."))
S.append(Paragraph("End state: screenshots that look colour-graded and lit — a lantern-lit world, not a tinted map.", goal))

# ---------------- WEEK 3 ----------------
S.append(Paragraph("Week 3 — The Isometric World (the big jump)", h1))
S.append(Paragraph("The one genuinely structural slice: tilt the world into 2.5D. Same procedural shapes, new "
                   "projection — buildings gain HEIGHT, castles TOWER, terrain gains depth. This is what moves the "
                   "look from 'diagram' to 'game'. It touches every draw call, the camera, town plans and the "
                   "cinematics — treat it like the camera slice: one shared projection function, built carefully.", body))
S.append(slice_item("V4.6", "The projection core",
    "One pure world_to_screen_iso((x, y, z), cam) used by EVERY draw call (diamond-tile ground projection, z for "
    "height). Draw order becomes painter's-algorithm by projected depth (back-to-front). Camera pan/zoom and the "
    "minimap carry over through the same transform. Terrain rebaked as diamond tiles with elevation shading."))
S.append(slice_item("V4.7", "Buildings with height",
    "Town-plan structures re-drawn as simple 3D forms: wall faces (lit side / shade side per the global light "
    "direction), roof planes, the castle keep and towers genuinely TALLER than houses, the palisade a ring of "
    "posts with height. Agents become upright figures standing ON the ground plane, shadows anchored at their feet."))
S.append(slice_item("V4.8", "Iso polish + cinematics re-seat",
    "War cinematics, speech bubbles, banners, crowns, trails, weather and the day/night light re-seated in the "
    "projection (bubbles float above heads; rain falls TOWARD the ground plane; territory drawn as ground overlay "
    "under buildings). Verify every prior feature reads correctly in iso before calling it done."))
S.append(Paragraph("End state: a tilted, deep, lit world — the screenshot people stop scrolling for.", goal))

# ---------------- WEEK 4 ----------------
S.append(Paragraph("Week 4 — Juice &amp; the Showcase Cut", h1))
S.append(slice_item("V4.9", "Juice",
    "Small motion that makes it feel alive: subtle screen shake on battle clashes, dust/particle burst when a "
    "building rises or falls, a gentle zoom-punch when the story banner fires, 1px squash-and-stretch on agent "
    "steps, banner cloth ripple. Each effect subtle; together they read as production value."))
S.append(slice_item("V4.10", "Showcase mode",
    "A --showcase flag: runs the most cinematic staged scenario, auto-glides the camera to major events (banner "
    "fires -> camera eases there), clean title overlay at start, HUD minimal. Built for screen recording — this is "
    "the mode the devlog footage comes from."))
S.append(Paragraph("End state: press record, get footage that looks like a real game trailer.", goal))

S.append(Spacer(1, 8))
S.append(rule())

# ---------------- SCHEDULE ----------------
S.append(Paragraph("Cadence", h1))
sched = [
    ["Week", "Theme", "Slices", "The visible win"],
    ["1", "Legibility & signal", "V4.1–V4.3", "Story readable from map + banner alone; bugs gone"],
    ["2", "Palette & light", "V4.4–V4.5", "Colour-graded frames; lantern-lit nights"],
    ["3", "Isometric world", "V4.6–V4.8", "2.5D depth — buildings tower, world tilts"],
    ["4", "Juice & showcase", "V4.9–V4.10", "Trailer-grade footage on demand"],
]
t = Table(sched, colWidths=[16*mm, 42*mm, 26*mm, doc.width - 84*mm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
    ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.8),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), LIGHT_BG]),
    ("GRID", (0, 0), (-1, -1), 0.4, RULE),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
S.append(t)
S.append(Spacer(1, 10))

S.append(Paragraph("Risk notes (stated up front)", h2))
for b in [
    "V4.6 (iso projection) is the heavy slice — a genuine rework touching every draw call. If it overruns, steal days from Week 4; juice is shrinkable, the projection is not.",
    "Palette and lighting are TASTE slices: budget explicit tuning evenings; screenshots before/after are the test, not the suite.",
    "Iso will surface layout bugs (overlaps, z-fights) the way the season slice surfaced the cache test — expect a fix-forward day after V4.7.",
    "Hard scope line: NO sprite art, no external assets, no 3D engine. The ceiling is crafted procedural 2.5D — which is enough to look genuinely cool.",
]:
    S.append(Paragraph(f"• {b}", bullet))

S.append(Paragraph("What this is for", h2))
S.append(Paragraph(
    "The simulation already earned its showcase — a world that produced Rex the Grasping, Kade the Liberator, and "
    "an uprising that recorded why it rose. This plan makes the picture worthy of the story: readable in week 1, "
    "crafted in week 2, dimensional in week 3, filmable in week 4. Then record the devlog from --showcase and "
    "publish.", body))

doc.build(S)
print("done")