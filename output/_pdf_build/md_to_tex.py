# -*- coding: utf-8 -*-
"""Convert project_report.md -> project_report.tex (pdflatex-ready)."""
import re, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
MD   = ROOT / "project_report.md"
TEX  = ROOT / "project_report.tex"
lines = MD.read_text(encoding="utf-8").split("\n")

NUL = "\x00"
UNI_PAIR = [("₉₅", r"\textsubscript{95}"), ("√T", r"$\sqrt{T}$")]
UNI = {
 "—":"---", "–":"--", "−":"-", "§":r"\S{}",
 "γ":r"$\gamma$", "α":r"$\alpha$", "η":r"$\eta$", "β":r"$\beta$",
 "ρ":r"$\rho$", "Δ":r"$\Delta$", "σ":r"$\sigma$", "λ":r"$\lambda$",
 "φ":r"$\phi$", "θ":r"$\theta$", "ω":r"$\omega$",
 "²":r"\textsuperscript{2}",
 "→":r"$\rightarrow$", "×":r"$\times$", "≈":r"$\approx$",
 "⇒":r"$\Rightarrow$", "≤":r"$\le$", "∈":r"$\in$",
 "è":r"\`{e}", "ï":"\\\"{\\i}", "√":r"$\surd$",
}

def texttt_escape(s):
    for a,b in [("\\",r"\textbackslash "),("_",r"\_"),("#",r"\#"),("%",r"\%"),
                ("&",r"\&"),("{",r"\{"),("}",r"\}"),("~",r"\textasciitilde "),
                ("^",r"\textasciicircum ")]:
        s = s.replace(a,b)
    return s

def apply_unicode(s):
    for a,b in UNI_PAIR: s = s.replace(a,b)
    for a,b in UNI.items(): s = s.replace(a,b)
    return s

def inline(s):
    # protect escaped markdown \* and \$
    s = s.replace(r"\*", NUL+"STAR"+NUL).replace(r"\$", NUL+"DOL"+NUL)
    # protect math
    store=[]
    def keep(m):
        store.append(m.group(0)); return NUL+f"M{len(store)-1}"+NUL
    s = re.sub(r"\$\$.+?\$\$", keep, s, flags=re.S)
    s = re.sub(r"\$[^$\n]+?\$", keep, s)
    # protect inline code
    codes=[]
    def kc(m):
        codes.append(m.group(1)); return NUL+f"C{len(codes)-1}"+NUL
    s = re.sub(r"`([^`]+)`", kc, s)
    # escape latex specials in remaining plain text
    for a,b in [("&",r"\&"),("%",r"\%"),("#",r"\#"),("_",r"\_"),("{",r"\{"),("}",r"\}"),
                ("~",r"$\sim$"),("<",r"$<$"),(">",r"$>$")]:
        s = s.replace(a,b)
    # emphasis
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"\*(.+?)\*", r"\\textit{\1}", s)
    # unicode/symbols
    s = apply_unicode(s)
    # restore
    for i,c in enumerate(codes):
        s = s.replace(NUL+f"C{i}"+NUL, r"\texttt{"+texttt_escape(c)+"}")
    for i,m in enumerate(store):
        s = s.replace(NUL+f"M{i}"+NUL, m)
    s = s.replace(NUL+"STAR"+NUL, "*").replace(NUL+"DOL"+NUL, r"\$")
    return s

def split_row(line):
    line = line.strip()
    if line.startswith("|"): line = line[1:]
    if line.endswith("|"): line = line[:-1]
    return [c.strip() for c in line.split("|")]

def render_table(tbl):
    header = split_row(tbl[0])
    rows   = [split_row(r) for r in tbl[2:]]
    n = len(header)
    head = " & ".join(r"\textbf{"+inline(c)+"}" for c in header) + r" \\"
    body = "\n".join(" & ".join(inline(c) for c in r) + r" \\" for r in rows)
    if "Implementation" in header:                      # 7-technique mapping
        spec = r"@{}c >{\raggedright\arraybackslash}p{3.0cm} X >{\raggedright\arraybackslash}p{1.6cm}@{}"
        env, w = "tabularx", r"{\textwidth}"
        return (f"\\begin{{center}}\\footnotesize\n\\begin{{{env}}}{w}{{{spec}}}\n"
                f"\\toprule\n{head}\n\\midrule\n{body}\n\\bottomrule\n"
                f"\\end{{{env}}}\n\\end{{center}}")
    if ("DM(h=5)" in header) or any("Constant-mean" in h for h in header):  # wide numeric
        spec = "l" + " Y"*(n-1)
        return (f"\\begin{{center}}\\footnotesize\n\\begin{{tabularx}}{{\\textwidth}}{{{spec}}}\n"
                f"\\toprule\n{head}\n\\midrule\n{body}\n\\bottomrule\n"
                f"\\end{{tabularx}}\n\\end{{center}}")
    spec = "l"*n                                        # small tables
    return (f"\\begin{{center}}\\small\n\\begin{{tabular}}{{{spec}}}\n"
            f"\\toprule\n{head}\n\\midrule\n{body}\n\\bottomrule\n"
            f"\\end{{tabular}}\n\\end{{center}}")

def is_tbl(l): return l.strip().startswith("|")

def render_blocks(ls):
    """Render a list of (dedented) markdown lines: paragraphs, tables, simple lists."""
    out=[]; i=0
    while i < len(ls):
        l = ls[i]
        if not l.strip():
            i+=1; continue
        if is_tbl(l):
            tbl=[]
            while i < len(ls) and is_tbl(ls[i]):
                tbl.append(ls[i]); i+=1
            out.append(render_table(tbl)); continue
        out.append(inline(l.strip())); i+=1
    return "\n\n".join(out)

# ---------- parse head ----------
src = "\n".join(lines)
body_start = src.index("\n## 1.")
head = lines[:src[:body_start].count("\n")+1]
rest = lines[src[:body_start].count("\n")+1:]

H1 = next(l[2:] for l in head if l.startswith("# "))
H2 = next(l[3:] for l in head if l.startswith("## "))
date_lines = [l for l in head if l.startswith("**Econometrics") or l.startswith("Politecnico")]
authors = next((l.split(":",1)[1].strip() for l in head if l.startswith("Authors")), "")
abstract = next((l for l in head if l.startswith("**Abstract.**")), "")
abstract = abstract.replace("**Abstract.**","").strip()

title = inline(H1) + r" \\[2pt] \large " + inline(H2)
date  = r" \\ ".join(inline(d.replace("**","")) for d in date_lines)

# ---------- parse body ----------
out=[]; i=0
def is_listline(l): return l.startswith("- ")
def indent_of(l): return len(l) - len(l.lstrip(" "))

while i < len(rest):
    l = rest[i]
    s = l.strip()
    if not s:
        i+=1; continue
    if s == "---":
        i+=1; continue
    if l.startswith("#### "):
        out.append(r"\subsubsection*{"+inline(l[5:].strip())+"}"); i+=1; continue
    if l.startswith("### "):
        out.append(r"\subsection*{"+inline(l[4:].strip())+"}"); i+=1; continue
    if l.startswith("## "):
        out.append(r"\section*{"+inline(l[3:].strip())+"}"); i+=1; continue
    if l.startswith("!["):                               # image + caption
        path = re.search(r"\(([^)]+)\)", l).group(1)
        j=i+1
        while j < len(rest) and not rest[j].strip(): j+=1
        cap = rest[j].lstrip("> ").strip() if j < len(rest) and rest[j].lstrip().startswith(">") else ""
        cap = re.sub(r"^\*\*Figure\s+\d+\.\*\*\s*", "", cap)
        out.append("\\begin{figure}[H]\\centering\n"
                   f"\\includegraphics[width=0.82\\linewidth,keepaspectratio]{{{path}}}\n"
                   f"\\caption{{{inline(cap)}}}\n\\end{{figure}}")
        i = j+1; continue
    if is_tbl(l):
        tbl=[]
        while i < len(rest) and is_tbl(rest[i]): tbl.append(rest[i]); i+=1
        out.append(render_table(tbl)); continue
    if is_listline(l):                                   # gather list group
        grp=[]
        while i < len(rest):
            cur=rest[i]
            if cur.strip()=="" :
                # peek: keep if following indented/list content belongs to group
                k=i+1
                while k < len(rest) and rest[k].strip()=="" : k+=1
                if k < len(rest) and (rest[k].startswith("  ") or is_listline(rest[k])):
                    grp.append(cur); i+=1; continue
                else:
                    break
            if is_listline(cur) or cur.startswith("  "):
                grp.append(cur); i+=1; continue
            break
        # strip trailing blanks
        while grp and grp[-1].strip()=="" : grp.pop()
        # split into items
        items=[]; cur=None
        for g in grp:
            if is_listline(g):
                if cur is not None: items.append(cur)
                cur=[g[2:]]                              # first line w/o "- "
            else:
                cur.append(g[2:] if g.startswith("  ") else g)
        if cur is not None: items.append(cur)
        complex_item = any(any(is_tbl(x) for x in it) for it in items)
        if complex_item:
            parts=[]
            for it in items:
                lead = it[0]
                nested = it[1:]
                block = render_blocks(nested)
                parts.append((r"\noindent " + inline(lead) + ("\n\n"+block if block.strip() else "")))
            out.append("\n\n".join(parts))
        else:
            body = "\n".join(r"\item "+inline(it[0].strip()) for it in items)
            out.append("\\begin{itemize}\\setlength{\\itemsep}{1pt}\n"+body+"\n\\end{itemize}")
        continue
    # plain paragraph
    out.append(inline(s)); i+=1

preamble = r"""% !TEX program = pdflatex
\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{float}
\usepackage{caption}
\usepackage[hidelinks]{hyperref}
\newcolumntype{Y}{>{\centering\arraybackslash}X}
\captionsetup{font=small,labelfont=bf}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3.5pt}
\graphicspath{{./}{./output/figures/}{./output/plots/}}
\title{@@TITLE@@}
\author{@@AUTHORS@@}
\date{@@DATE@@}
\begin{document}
\maketitle
\vspace{-1.2em}
\begin{abstract}\noindent @@ABSTRACT@@ \end{abstract}
"""
preamble = (preamble.replace("@@TITLE@@", title)
                    .replace("@@AUTHORS@@", inline(authors))
                    .replace("@@DATE@@", date)
                    .replace("@@ABSTRACT@@", inline(abstract)))

TEX.write_text(preamble + "\n\n" + "\n\n".join(out) + "\n\n\\end{document}\n", encoding="utf-8")
print("wrote", TEX, "| blocks:", len(out))
