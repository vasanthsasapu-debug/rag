"""
PDF Text Extraction Module
Extracts clean text from arXiv PDFs using PyMuPDF (fitz).
Handles headers, footers, references, and multi-column layouts.
"""

import fitz  # PyMuPDF
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ExtractedDocument:
    """Represents a processed document ready for chunking."""
    doc_id: str
    title: str
    authors: list[str]
    abstract: str
    full_text: str
    sections: list[dict]      # [{"heading": "...", "content": "..."}, ...]
    page_count: int
    source_file: str
    metadata: dict             # Original metadata from paper_metadata.json


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    """
    Extract clean text from a PDF using PyMuPDF.
    Handles multi-column layouts common in arXiv papers.

    Returns:
        Tuple of (full_text, page_count)
    """
    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    full_text = []

    for page_num, page in enumerate(doc):
        # Extract text with layout preservation
        # sort=True reorders text blocks by position (handles multi-column)
        text = page.get_text("text", sort=True)

        # Clean up common arXiv PDF artifacts
        text = _clean_page_text(text, page_num, page_count)
        if text.strip():
            full_text.append(text)

    doc.close()
    return "\n\n".join(full_text), page_count


def _clean_page_text(text: str, page_num: int, total_pages: int) -> str:
    """Clean extracted text from common PDF artifacts."""
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines (but preserve paragraph breaks)
        if not stripped:
            cleaned_lines.append("")
            continue

        # Skip page numbers (standalone numbers, "Page X", "X of Y")
        if re.match(r"^\d{1,3}$", stripped):
            continue
        if re.match(r"^(page\s+)?\d{1,3}\s*(of\s+\d{1,3})?$", stripped, re.IGNORECASE):
            continue

        # Skip common arXiv headers/footers
        if re.match(r"^(preprint|under review|arxiv:\d)", stripped, re.IGNORECASE):
            continue

        # Skip lines that are just URLs
        if re.match(r"^https?://\S+$", stripped):
            continue

        # Skip figure and table captions (e.g., "Figure 2. ...", "Table 1: ...")
        if re.match(r"^(Figure|Fig\.|Table)\s+\d+[.:]\s", stripped, re.IGNORECASE):
            continue

        # Skip lines that look like matrix/table data (mostly numbers and +/- signs)
        # e.g., "+1.2 −0.2 −2.4 −3.4" from extracted tables/figures
        # Only apply to lines longer than 15 chars to avoid stripping short content
        # BUT preserve lines that start with a section heading pattern
        if len(stripped) > 15:
            # Don't filter if line starts with what looks like a section heading
            looks_like_heading = re.match(
                r"^\d+\.?\d*\s{1,4}[A-Z]{2,}", stripped
            )
            if not looks_like_heading:
                non_numeric = re.sub(r"[\d\s\.\+\-\−×,%()\u2212]", "", stripped)
                if len(non_numeric) < len(stripped) * 0.15:
                    continue

        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix hyphenated line breaks (common in PDFs)
    # Only rejoin if the result looks like a word continuation (lowercase after hyphen)
    # This avoids merging legitimate hyphens like "state-\nof-the-art"
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)

    return text


def _split_multicolumn_lines(text: str) -> str:
    """
    Pre-process text to split lines merged from multi-column PDF layouts.
    PyMuPDF with sort=True often puts left-column and right-column text on
    the same line, separated by large whitespace gaps (5+ spaces).
    Splitting on these gaps puts headings on their own line for regex matching.
    """
    processed_lines = []
    for line in text.split("\n"):
        # Split on 5+ consecutive spaces (column boundary indicator)
        parts = re.split(r"\s{5,}", line)
        for part in parts:
            stripped = part.strip()
            if stripped:
                processed_lines.append(stripped)
            else:
                processed_lines.append("")
    return "\n".join(processed_lines)


def _find_heading_position(full_text: str, section_num: str, heading_text: str) -> int:
    """
    Find the position of a section heading in the original (unprocessed) text.
    Handles cases where the heading may have trailing junk from multi-column merging.
    """
    # Try matching section number + first 20 chars of heading with flexible whitespace
    escaped_prefix = re.escape(heading_text[:min(20, len(heading_text))])
    pattern = re.compile(
        r"^" + re.escape(section_num) + r"\s{1,6}" + escaped_prefix,
        re.MULTILINE
    )
    match = pattern.search(full_text)
    if match:
        return match.start()

    # Fallback: just the section number + first word
    first_word = heading_text.split()[0] if heading_text.split() else ""
    if first_word:
        pattern2 = re.compile(
            r"^" + re.escape(section_num) + r"\s{1,6}" + re.escape(first_word),
            re.MULTILINE
        )
        match2 = pattern2.search(full_text)
        if match2:
            return match2.start()

    return -1


def extract_sections(full_text: str) -> list[dict]:
    """
    Split document text into sections based on heading patterns.
    arXiv papers use various heading styles:
      - Numbered title case: "1 Introduction", "2.1 Method"
      - Numbered ALL CAPS: "1 INTRODUCTION", "3 AWQ: ACTIVATION-AWARE WEIGHT"
      - Unnumbered: "Abstract", "ABSTRACT", "Conclusion"

    Uses a two-phase approach:
      1. Pre-process text to split multi-column merged lines
      2. Match heading patterns on the cleaned text
      3. Map matches back to original text for content extraction
    """
    # Phase 1: Split multi-column merged lines so headings land on their own line
    processed_text = _split_multicolumn_lines(full_text)

    # Phase 2: Match section headings on processed text
    # Numbered sections (title case, ALL CAPS, or mixed like "LLM as a Judge")
    # Pattern: "1 Introduction", "2.1 Method", "3 AWQ: ACTIVATION-AWARE..."
    # NOT: "1. Bowing" (list items) - the (?!\.) prevents matching "1." as a section num
    section_pattern = re.compile(
        r"^(\d+(?:\.\d+)*)(?!\.)\s{1,4}([A-Z][A-Za-z\s:,&\-\u2019\u2018()]+)\s*$",
        re.MULTILINE
    )

    # Unnumbered major headings
    unnumbered_pattern = re.compile(
        r"^(ABSTRACT|Abstract|INTRODUCTION|Introduction|RELATED WORK|Related Work|"
        r"METHODOLOGY|Methodology|METHOD|Method|METHODS|Methods|"
        r"APPROACH|Approach|EXPERIMENTS?|Experiments?|RESULTS?|Results?|"
        r"DISCUSSION|Discussion|CONCLUSION|Conclusion|CONCLUSIONS|Conclusions|"
        r"REFERENCES|References|ACKNOWLEDGEMENTS?|Acknowledgements?|Acknowledgement|"
        r"APPENDIX|Appendix|LIMITATIONS|Limitations|FUTURE WORK|Future Work)\s*$",
        re.MULTILINE
    )

    sections = []
    split_points = []

    # Find numbered section headings
    for match in section_pattern.finditer(processed_text):
        section_num = match.group(1)

        # Validate: real section numbers are 1-20 (filters table data like "14 GPTQ")
        try:
            first_num = int(section_num.split(".")[0])
            if first_num > 20:
                continue
        except ValueError:
            continue

        heading_text = match.group(2).strip()

        # Skip too short (noise) or too long (sentences)
        if len(heading_text) < 3 or len(heading_text) > 80:
            continue

        # Map back to position in original text
        orig_pos = _find_heading_position(full_text, section_num, heading_text)
        if orig_pos >= 0:
            split_points.append({
                "start": orig_pos,
                "heading": f"{section_num} {heading_text}",
            })

    # Find unnumbered headings
    for match in unnumbered_pattern.finditer(processed_text):
        heading_text = match.group(0).strip()
        # Find in original text (search from beginning each time)
        orig_pos = full_text.find(heading_text)
        if orig_pos >= 0:
            split_points.append({
                "start": orig_pos,
                "heading": heading_text,
            })

    # Sort by position in text
    split_points.sort(key=lambda x: x["start"])

    # Deduplicate: if two headings are within 10 chars of each other, keep the first
    if len(split_points) > 1:
        deduped = [split_points[0]]
        for sp in split_points[1:]:
            if sp["start"] - deduped[-1]["start"] > 10:
                deduped.append(sp)
        split_points = deduped

    # If no sections found, return the whole text as one section
    if not split_points:
        return [{"heading": "Full Document", "content": full_text.strip()}]

    # Add text before first section heading (title, authors, affiliations)
    if split_points[0]["start"] > 200:
        preamble = full_text[: split_points[0]["start"]].strip()
        if preamble:
            sections.append({"heading": "Preamble", "content": preamble})

    # Extract content between headings from ORIGINAL text
    for i, point in enumerate(split_points):
        start = point["start"]
        end = split_points[i + 1]["start"] if i + 1 < len(split_points) else len(full_text)
        content = full_text[start:end].strip()

        # Remove the heading line itself from the content
        content_lines = content.split("\n", 1)
        content_body = content_lines[1].strip() if len(content_lines) > 1 else ""

        # Skip references and acknowledgement sections (noise for RAG)
        heading_lower = point["heading"].lower()
        if "reference" in heading_lower:
            continue
        if "acknowledgement" in heading_lower or "acknowledgment" in heading_lower:
            continue

        if content_body:
            sections.append({
                "heading": point["heading"],
                "content": content_body,
            })

    return sections


def extract_abstract(full_text: str) -> str:
    """Try to extract the abstract from the paper text."""
    # Pattern 1: Explicit "Abstract" heading
    abstract_match = re.search(
        r"(?:^|\n)\s*Abstract\s*\n(.*?)(?=\n\s*(?:\d+\.?\s+)?(?:Introduction|1\s))",
        full_text,
        re.DOTALL | re.IGNORECASE,
    )
    if abstract_match:
        return abstract_match.group(1).strip()

    # Pattern 2: First substantial paragraph (common in arXiv)
    paragraphs = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 100]
    if paragraphs:
        # Abstract is usually the first long paragraph
        return paragraphs[0][:2000]  # Cap at 2000 chars

    return ""


def process_single_pdf(
    pdf_path: Path,
    metadata: Optional[dict] = None,
) -> ExtractedDocument:
    """Process a single PDF into an ExtractedDocument."""
    # Extract raw text and page count in a single open
    full_text, page_count = extract_text_from_pdf(pdf_path)

    # Parse sections
    sections = extract_sections(full_text)

    # Get abstract from metadata or extract from text
    abstract = ""
    if metadata and metadata.get("abstract"):
        abstract = metadata["abstract"]
    else:
        abstract = extract_abstract(full_text)

    # Build document
    doc = ExtractedDocument(
        doc_id=metadata.get("arxiv_id", pdf_path.stem) if metadata else pdf_path.stem,
        title=metadata.get("title", pdf_path.stem) if metadata else pdf_path.stem,
        authors=metadata.get("authors", []) if metadata else [],
        abstract=abstract,
        full_text=full_text,
        sections=sections,
        page_count=page_count,
        source_file=pdf_path.name,
        metadata=metadata or {},
    )
    return doc


def process_all_pdfs(
    pdf_dir: Path,
    metadata_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
) -> list[ExtractedDocument]:
    """
    Process all PDFs in a directory into ExtractedDocuments.

    Args:
        pdf_dir: Directory containing PDFs
        metadata_file: Optional paper_metadata.json for enrichment
        output_file: Optional path to save processed documents as JSON
    """
    # Load metadata if available
    metadata_lookup = {}
    if metadata_file and metadata_file.exists():
        with open(metadata_file) as f:
            metadata_list = json.load(f)
        metadata_lookup = {m["pdf_filename"]: m for m in metadata_list}
        print(f"📋 Loaded metadata for {len(metadata_lookup)} papers")

    # Find all PDFs
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"❌ No PDFs found in {pdf_dir}")
        return []

    print(f"📄 Processing {len(pdfs)} PDFs...\n")

    documents = []
    failed = []

    for i, pdf_path in enumerate(pdfs):
        try:
            metadata = metadata_lookup.get(pdf_path.name)
            doc = process_single_pdf(pdf_path, metadata)
            documents.append(doc)

            title_display = doc.title[:60] + "..." if len(doc.title) > 60 else doc.title
            print(f"  ✅ [{i+1}/{len(pdfs)}] {title_display}")
            print(f"     → {doc.page_count} pages, {len(doc.sections)} sections, "
                  f"{len(doc.full_text):,} chars")

        except Exception as e:
            failed.append((pdf_path.name, str(e)))
            print(f"  ❌ [{i+1}/{len(pdfs)}] {pdf_path.name}: {e}")

    # Save processed documents
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        docs_json = [asdict(d) for d in documents]
        with open(output_file, "w") as f:
            json.dump(docs_json, f, indent=2)
        print(f"\n💾 Saved {len(documents)} documents to {output_file}")

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Processed: {len(documents)} documents")
    if failed:
        print(f"❌ Failed: {len(failed)} documents")
        for name, err in failed:
            print(f"   • {name}: {err}")
    total_chars = sum(len(d.full_text) for d in documents)
    total_sections = sum(len(d.sections) for d in documents)
    print(f"📊 Total: {total_chars:,} characters, {total_sections} sections")
    print(f"{'='*60}")

    return documents


if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
    PDF_DIR = DATA_DIR / "raw_pdfs"
    METADATA_FILE = DATA_DIR / "paper_metadata.json"
    OUTPUT_FILE = DATA_DIR / "processed" / "extracted_documents.json"

    documents = process_all_pdfs(PDF_DIR, METADATA_FILE, OUTPUT_FILE)
