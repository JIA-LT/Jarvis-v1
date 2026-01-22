import re
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+\s*)?$")  # supports "## Title ###"
_CODE_FENCE_RE = re.compile(r"^\s*```")  # fenced code blocks


@dataclass
class Heading:
    level: int
    text: str


def _stable_id(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def _header_path(stack: List[Heading], max_level: int = 3) -> str:
    """Build 'H1 > H2 > H3' style path, keeping up to max_level headings (H1..Hmax)."""
    filtered = [h for h in stack if h.level <= max_level]
    return " > ".join([f"H{h.level} {h.text}" for h in filtered]) if filtered else ""


def chunk_text(
    md_text: str,
    *,
    max_chars: int = 1800,
    min_chars: int = 250,
    max_heading_level_for_path: int = 3,
    keep_code_blocks_intact: bool = True,
) -> List[Dict]:
    """
    Chunk Markdown by heading structure (H1/H2/H3...) + paragraph boundaries.

    Returns list of dict chunks:
      {
        "chunk_id": str,
        "content": str,
        "header_path": str,     # e.g., "H1 Intro > H2 Details"
        "start_line": int,      # 1-based inclusive
        "end_line": int         # 1-based inclusive
      }

    Design goals:
    - Stable, reproducible chunk boundaries (good for RAG).
    - Respect code fences (optionally keep intact).
    - Avoid tiny fragments by merging up to min_chars when possible.
    """
    lines = md_text.splitlines()
    n_lines = len(lines)

    # Track code fences so we don't treat headings inside code blocks as headings.
    in_code = False

    heading_stack: List[Heading] = []
    blocks: List[Tuple[int, int, str, str]] = []
    # blocks: (start_line, end_line, header_path, text)

    # Current section accumulator (within current heading path)
    sec_start = 1
    sec_lines: List[str] = []

    def flush_section(end_line_inclusive: int):
        nonlocal sec_start, sec_lines
        if not sec_lines:
            sec_start = end_line_inclusive + 1
            return
        hp = _header_path(heading_stack, max_level=max_heading_level_for_path)
        text = "\n".join(sec_lines).strip("\n")
        if text.strip():
            blocks.append((sec_start, end_line_inclusive, hp, text))
        sec_lines = []
        sec_start = end_line_inclusive + 1

    # Pass 1: split into "sections" at headings (outside code fences)
    for i, raw in enumerate(lines, start=1):
        if keep_code_blocks_intact and _CODE_FENCE_RE.match(raw):
            in_code = not in_code

        m = None if in_code else _HEADING_RE.match(raw)

        if m:
            # New heading begins; flush content before this heading
            flush_section(i - 1)

            level = len(m.group(1))
            text = m.group(2).strip()

            # Update heading stack: pop >= level, push current
            while heading_stack and heading_stack[-1].level >= level:
                heading_stack.pop()
            heading_stack.append(Heading(level=level, text=text))

            # Headings themselves should belong to the following content, not a standalone chunk.
            # But we DO want heading line included (often helpful for retrieval).
            sec_start = i
            sec_lines = [raw]
        else:
            sec_lines.append(raw)

    flush_section(n_lines)

    # Pass 2: within each section block, split by paragraphs to respect max_chars,
    # while keeping code fences intact.
    def split_section_to_chunks(
        start_line: int, end_line: int, hp: str, text: str
    ) -> List[Tuple[int, int, str, str]]:
        sec = lines[start_line - 1 : end_line]  # original lines for accurate line numbers
        chunks: List[Tuple[int, int, str, str]] = []

        # Build paragraph units with line ranges, respecting code fences.
        units: List[Tuple[int, int, str]] = []  # (u_start, u_end, u_text)
        u_start = start_line
        buf: List[str] = []
        in_code_local = False

        def flush_unit(u_end: int):
            nonlocal u_start, buf
            if buf:
                units.append((u_start, u_end, "\n".join(buf).rstrip("\n")))
                buf = []
            u_start = u_end + 1

        for idx, raw in enumerate(sec, start=start_line):
            if keep_code_blocks_intact and _CODE_FENCE_RE.match(raw):
                in_code_local = not in_code_local
                buf.append(raw)
                continue

            # Paragraph boundary: blank line outside code
            if (not in_code_local) and (raw.strip() == ""):
                buf.append(raw)
                flush_unit(idx)
            else:
                buf.append(raw)

        flush_unit(end_line)

        # Merge units into chunks by max_chars
        cur: List[str] = []
        cur_start: Optional[int] = None
        cur_end: Optional[int] = None
        cur_len = 0

        def flush_chunk():
            nonlocal cur, cur_start, cur_end, cur_len
            if cur and cur_start is not None and cur_end is not None:
                t = "\n".join(cur).strip("\n")
                if t.strip():
                    chunks.append((cur_start, cur_end, hp, t))
            cur, cur_start, cur_end, cur_len = [], None, None, 0

        for u_s, u_e, u_text in units:
            u_clean = u_text.strip("\n")
            if not u_clean.strip():
                continue
            u_len = len(u_clean)

            # If unit itself is huge (e.g., big code block), keep as its own chunk
            if u_len > max_chars:
                flush_chunk()
                chunks.append((u_s, u_e, hp, u_clean))
                continue

            if cur_start is None:
                cur_start, cur_end = u_s, u_e
                cur = [u_clean]
                cur_len = u_len
                continue

            # If adding unit would exceed max_chars, flush current and start new
            if cur_len + 2 + u_len > max_chars:
                flush_chunk()
                cur_start, cur_end = u_s, u_e
                cur = [u_clean]
                cur_len = u_len
            else:
                cur.append(u_clean)
                cur_end = u_e
                cur_len += 2 + u_len

        flush_chunk()

        # Pass 3: merge too-small chunks with neighbors when possible (same hp)
        merged: List[Tuple[int, int, str, str]] = []
        for c in chunks:
            if not merged:
                merged.append(c)
                continue
            prev_s, prev_e, prev_hp, prev_text = merged[-1]
            c_s, c_e, c_hp, c_text = c
            if prev_hp == c_hp and len(prev_text) < min_chars:
                # merge prev into current (or merge current into prev) based on size
                new_text = (prev_text + "\n\n" + c_text).strip("\n")
                if len(new_text) <= max_chars:
                    merged[-1] = (prev_s, c_e, prev_hp, new_text)
                else:
                    merged.append(c)
            else:
                merged.append(c)

        return merged

    final_chunks: List[Tuple[int, int, str, str]] = []
    for s_line, e_line, hp, text in blocks:
        final_chunks.extend(split_section_to_chunks(s_line, e_line, hp, text))

    # Build output dicts with stable chunk_id
    out: List[Dict] = []
    for s_line, e_line, hp, text in final_chunks:
        # Stable ID based on header_path + line range + content hash
        base = f"{hp}::{s_line}-{e_line}::{_stable_id(text, 16)}"
        out.append(
            {
                "chunk_id": _stable_id(base, 16),
                "content": text,
                "header_path": hp,
                "start_line": s_line,
                "end_line": e_line,
            }
        )

    return out


