import os
import re
import torch
from collections import Counter
from flask import Flask, render_template, request, send_file
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from transformers import pipeline
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

# ==============================
# PATH SETUP
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
FONT_FOLDER = os.path.join(BASE_DIR, "fonts")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==============================
# REGISTER FONTS
# ==============================

mal_font = os.path.join(FONT_FOLDER, "NotoSansMalayalam-Regular.ttf")
hin_font = os.path.join(FONT_FOLDER, "NotoSansDevanagari-Regular.ttf")

if os.path.exists(mal_font):
    pdfmetrics.registerFont(TTFont("MalayalamFont", mal_font))

if os.path.exists(hin_font):
    pdfmetrics.registerFont(TTFont("HindiFont", hin_font))

# ==============================
# FLASK APP
# ==============================

app = Flask(__name__)
torch.set_num_threads(2)

# ==============================
# LAZY MODEL LOADING
# ==============================

whisper_model = None
summarizer = None
flan_model = None

# ==============================
# FILTERS
# ==============================

FILLER_PATTERNS = [
    r"\bokay\b", r"\bok\b", r"\bso\b", r"\bnow\b", r"\bwell\b", r"\bright\b",
    r"\byou know\b", r"\bsee this\b", r"\blook at this\b", r"\bwait for this example\b",
    r"\bif you don't understand\b", r"\blet us see\b", r"\bwe will see\b",
    r"\bnext we will see\b"
]

COURSE_META_PATTERNS = [
    r"\bthis course\b", r"\bundergraduate students\b", r"\bgate exams\b",
    r"\bccna\b", r"\bprerequisite course\b", r"\bupcoming lectures\b",
    r"\bsyllabi\b", r"\bthis lecture series\b", r"\bprepare for\b"
]

STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
    "this", "that", "these", "those", "it", "as", "by", "are", "was", "were",
    "be", "been", "being", "at", "from", "we", "you", "they", "he", "she",
    "them", "his", "her", "their", "our", "your", "can", "may", "will", "would",
    "should", "could", "about", "into", "than", "then", "also", "such", "if",
    "but", "so", "because", "when", "where", "which", "who", "whom", "what",
    "how", "why", "all", "any", "some", "many", "much", "more", "most", "other",
    "another", "very", "there", "here", "have", "has", "had", "do", "does",
    "did", "not", "only", "just"
}

TOPIC_KEYWORDS = [
    "network", "computer network", "node", "nodes", "link", "links",
    "wired", "wireless", "router", "switch", "bridge", "communication",
    "end device", "end devices", "intermediary device", "intermediary devices",
    "server", "computer", "data", "device", "devices"
]

# ==============================
# MODEL LOADERS
# ==============================

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model...")
        whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )
    return whisper_model


def get_summarizer():
    global summarizer
    if summarizer is None:
        print("Loading BART summarizer...")
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
    return summarizer


def get_flan_model():
    global flan_model
    if flan_model is None:
        print("Loading FLAN-T5 model...")
        flan_model = pipeline(
            "text2text-generation",
            model="google/flan-t5-small"
        )
    return flan_model

# ==============================
# TEXT HELPERS
# ==============================

def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def format_timestamp(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def contains_pattern(sentence, patterns):
    lowered = sentence.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def has_topic_keyword(sentence):
    lowered = sentence.lower()
    return any(keyword in lowered for keyword in TOPIC_KEYWORDS)


def is_meaningful_sentence(sentence):
    s = sentence.strip()

    if not s:
        return False

    if len(s.split()) < 5:
        return False

    if contains_pattern(s, FILLER_PATTERNS):
        return False

    if contains_pattern(s, COURSE_META_PATTERNS):
        return False

    if s.endswith(("where", "when", "which", "that", "because")):
        return False

    bad_starts = ("and ", "but ", "so ", "now ", "okay ", "ok ", "see ", "wait ")
    if s.lower().startswith(bad_starts) and len(s.split()) < 10:
        return False

    return True


def remove_duplicate_sentences(sentences):
    seen = set()
    cleaned = []

    for s in sentences:
        norm = s.strip().lower()
        norm = re.sub(r"[^a-z0-9\s]", "", norm)
        norm = normalize_spaces(norm)

        if norm and norm not in seen:
            cleaned.append(s.strip())
            seen.add(norm)

    return cleaned

# ==============================
# CLEAN TRANSCRIPT
# ==============================

def clean_transcript(text):
    text = normalize_spaces(text)
    sentences = split_sentences(text)

    cleaned_sentences = []

    for s in sentences:
        s = re.sub(r'\b(\w+)( \1\b)+', r'\1', s, flags=re.IGNORECASE)
        s = normalize_spaces(s)

        if is_meaningful_sentence(s):
            cleaned_sentences.append(s)

    cleaned_sentences = remove_duplicate_sentences(cleaned_sentences)

    topic_sentences = [s for s in cleaned_sentences if has_topic_keyword(s)]
    other_sentences = [s for s in cleaned_sentences if not has_topic_keyword(s)]

    return " ".join(topic_sentences + other_sentences)

# ==============================
# CHUNK HELPERS
# ==============================

def chunk_text(text, chunk_size=350):
    words = text.split()

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])

        if len(chunk) > 2000:
            chunk = chunk[:2000]

        yield chunk


def combine_segments_to_text(segments):
    return " ".join(seg["text"].strip() for seg in segments if seg["text"].strip())


def chunk_segments_by_word_limit(segments, max_words=180):
    grouped = []
    current_group = []
    current_word_count = 0

    for seg in segments:
        wc = len(seg["text"].split())

        if current_group and current_word_count + wc > max_words:
            grouped.append(current_group)
            current_group = []
            current_word_count = 0

        current_group.append(seg)
        current_word_count += wc

    if current_group:
        grouped.append(current_group)

    return grouped

# ==============================
# DEFINITIONS / GLOSSARY
# ==============================

def detect_definitions(text):
    sentences = split_sentences(text)
    patterns = [
        r"\bis called\b", r"\bis known as\b", r"\bis defined as\b",
        r"\brefers to\b", r"\bis a\b", r"\bis an\b"
    ]

    definitions = []
    for sentence in sentences:
        if any(re.search(p, sentence.lower()) for p in patterns):
            if is_meaningful_sentence(sentence):
                definitions.append(sentence)

    return remove_duplicate_sentences(definitions)[:10]


def generate_glossary(definitions, cleaned_text):
    glossary = []

    for d in definitions[:8]:
        parts = re.split(
            r"\bis called\b|\bis known as\b|\bis defined as\b|\brefers to\b|\bis a\b|\bis an\b",
            d,
            maxsplit=1,
            flags=re.IGNORECASE
        )

        if len(parts) >= 2:
            term = parts[0].strip().rstrip(",")
            meaning = parts[1].strip()

            if 1 <= len(term.split()) <= 5 and len(meaning.split()) >= 3:
                glossary.append((term, meaning))

    if len(glossary) < 5:
        sentences = split_sentences(cleaned_text)
        for sentence in sentences:
            if is_meaningful_sentence(sentence) and has_topic_keyword(sentence):
                words = sentence.split()
                if words:
                    term = words[0].strip(",.:;").title()
                    glossary.append((term, sentence))
            if len(glossary) >= 8:
                break

    # deduplicate glossary terms
    final_glossary = []
    seen_terms = set()

    for term, meaning in glossary:
        term_norm = term.lower().strip()
        if term_norm not in seen_terms:
            final_glossary.append((term, meaning))
            seen_terms.add(term_norm)

    return final_glossary[:8]

# ==============================
# SUMMARIZATION
# ==============================

def filter_summary_sentences(text):
    sentences = split_sentences(text)
    good = []

    for s in sentences:
        if is_meaningful_sentence(s) or has_topic_keyword(s):
            good.append(s)

    return " ".join(remove_duplicate_sentences(good))


def summarize_text(text):
    summarizer_model = get_summarizer()

    if len(text.split()) < 120:
        return text

    chunk_summaries = []

    for chunk in chunk_text(text, chunk_size=350):
        safe_chunk = chunk

        if len(safe_chunk.split()) > 350:
            safe_chunk = " ".join(safe_chunk.split()[:350])

        result = summarizer_model(
            safe_chunk,
            max_length=200,
            min_length=80,
            do_sample=False
        )

        chunk_summary = filter_summary_sentences(result[0]["summary_text"])
        chunk_summaries.append(chunk_summary)

    combined = " ".join(chunk_summaries)

    if len(combined.split()) > 600:
        combined = " ".join(combined.split()[:600])

    final_result = summarizer_model(
        combined,
        max_length=350,
        min_length=150,
        do_sample=False
    )

    return filter_summary_sentences(final_result[0]["summary_text"])


def summarize_topic_chunks(timestamped_chunks):
    summarizer_model = get_summarizer()
    topic_summaries = []

    for chunk in timestamped_chunks:
        content = chunk["text"]

        if len(content.split()) < 40:
            topic_summary = content
        else:
            safe_content = content
            if len(safe_content.split()) > 350:
                safe_content = " ".join(safe_content.split()[:350])

            result = summarizer_model(
                safe_content,
                max_length=180,
                min_length=80,
                do_sample=False
            )
            topic_summary = result[0]["summary_text"]

        topic_summaries.append({
            "start": chunk["start"],
            "end": chunk["end"],
            "summary": filter_summary_sentences(topic_summary)
        })

    return topic_summaries

# ==============================
# FLAN-T5 BASED TITLE / IMPORTANT NOTES / MCQ / FLASHCARD
# ==============================

def run_flan(prompt, max_length=512):
    model = get_flan_model()
    result = model(prompt, max_length=max_length, do_sample=False)
    return result[0]["generated_text"].strip()


def fallback_title(text):
    lowered = text.lower()

    if "computer network" in lowered:
        return "Introduction to Computer Networks"
    if "network" in lowered and "node" in lowered:
        return "Computer Networks: Nodes and Links"
    if "end devices" in lowered or "intermediary devices" in lowered:
        return "Network Devices and Communication"

    return "Lecture Notes"


def generate_topic_title(text):
    prompt = f"""
Generate a clear and specific academic title for this lecture.

Rules:
- 4 to 8 words
- Must describe the topic clearly
- No generic names like Lecture Notes

Lecture:
{text}
"""
    title = run_flan(prompt, max_length=20)

    if not title or len(title.split()) < 3:
        return fallback_title(text)

    return title


def generate_important_notes(summary_text):
    prompt = f"""
Convert the following lecture summary into high-quality study notes.

Rules:
1. Write ONLY important points
2. Use clear bullet points
3. Each point must be meaningful, accurate, and informative
4. Avoid repetition
5. Avoid vague sentences
6. Include technical explanations when possible
7. Cover all key concepts properly
8. Do not create sections like Introduction, Key Concepts, or Conclusion

Output format:
- point 1
- point 2
- point 3

Lecture Summary:
{summary_text}
"""

    notes = run_flan(prompt, max_length=700)

    if "-" not in notes:
        sentences = split_sentences(summary_text)
        filtered = remove_duplicate_sentences([
            s for s in sentences if is_meaningful_sentence(s) or has_topic_keyword(s)
        ])[:10]
        notes = "\n".join([f"- {s}" for s in filtered])

    return notes


def merge_important_notes_and_definitions(important_notes, definitions):
    note_lines = []

    for line in important_notes.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("-"):
            note_lines.append(line[1:].strip())
        else:
            note_lines.append(line)

    merged = note_lines + definitions
    merged = remove_duplicate_sentences([x for x in merged if x.strip()])

    return "\n".join([f"- {line}" for line in merged])


def generate_mcqs(source_text):
    prompt = f"""
Generate exactly 8 multiple choice questions from the following lecture content.

Rules:
1. Return exactly 8 questions.
2. Each question must be different.
3. Each question must have 4 options: A, B, C, D.
4. Give the correct answer after each question.
5. Use only the lecture content.
6. Do not stop after one question.

Format exactly:

Q1. ...
A) ...
B) ...
C) ...
D) ...
Answer: ...

Q2. ...
A) ...
B) ...
C) ...
D) ...
Answer: ...

Lecture Content:
{source_text}
"""
    return run_flan(prompt, max_length=1200)


def generate_flashcards(source_text):
    prompt = f"""
Generate exactly 8 study flashcards from the following lecture content.

Rules:
1. Return exactly 8 flashcards.
2. Each flashcard must contain one question and one answer.
3. Keep answers short and meaningful.
4. Use only the lecture content.
5. Do not stop after one flashcard.

Format exactly:

Q: ...
A: ...

Q: ...
A: ...

Lecture Content:
{source_text}
"""
    return run_flan(prompt, max_length=900)

# ==============================
# TIMESTAMPS
# ==============================

def build_timestamped_notes(topic_summaries):
    lines = []

    for item in topic_summaries:
        start_ts = format_timestamp(item["start"])
        end_ts = format_timestamp(item["end"])
        lines.append(f"- [{start_ts} - {end_ts}] {item['summary']}")

    return lines

# ==============================
# TRANSLATION
# ==============================

def translate_bullet_text_block(text, language):
    if language == "en":
        return text

    lines = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            lines.append("")
            continue

        if line.startswith("-"):
            content = line[1:].strip()
            translated = GoogleTranslator(source="auto", target=language).translate(content)
            lines.append("- " + translated)
        else:
            translated = GoogleTranslator(source="auto", target=language).translate(line)
            lines.append(translated)

    return "\n".join(lines)


def translate_text_block(text, language):
    if language == "en":
        return text

    lines = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            lines.append("")
            continue

        translated = GoogleTranslator(source="auto", target=language).translate(line)
        lines.append(translated)

    return "\n".join(lines)


def translate_bullet_lines(lines, language):
    if language == "en":
        return lines

    translated = []

    for line in lines:
        line = line.strip()

        if not line:
            translated.append("")
            continue

        bullet_prefix = ""
        content = line

        if line.startswith("- "):
            bullet_prefix = "- "
            content = line[2:].strip()

        translated_text = GoogleTranslator(source="auto", target=language).translate(content)
        translated.append(bullet_prefix + translated_text)

    return translated


def translate_glossary(glossary, language):
    if language == "en":
        return glossary

    translated = []
    for term, meaning in glossary:
        translated_term = GoogleTranslator(source="auto", target=language).translate(term)
        translated_meaning = GoogleTranslator(source="auto", target=language).translate(meaning)
        translated.append((translated_term, translated_meaning))

    return translated

# ==============================
# PDF WATERMARK
# ==============================

def add_watermark(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillGray(0.85)
    canvas.drawString(40, 20, "SMART LECTURE NOTES")
    canvas.restoreState()

# ==============================
# PDF GENERATION
# ==============================

def get_font_name(language):
    if language == "ml" and "MalayalamFont" in pdfmetrics.getRegisteredFontNames():
        return "MalayalamFont"
    if language == "hi" and "HindiFont" in pdfmetrics.getRegisteredFontNames():
        return "HindiFont"
    return "Helvetica"


def generate_pdf(topic_title, important_points, language, glossary, timestamped_notes, mcqs, flashcards):
    pdf_path = os.path.join(OUTPUT_FOLDER, "Lecture_Notes.pdf")
    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    title_style = styles["Heading1"]
    heading_style = styles["Heading2"]
    normal_style = styles["Normal"]

    font_name = get_font_name(language)
    title_style.fontName = font_name
    heading_style.fontName = font_name
    normal_style.fontName = font_name

    elements = []

    elements.append(Paragraph(topic_title, title_style))
    elements.append(Spacer(1, 20))

    # Important Points
    elements.append(Paragraph("Important Points", heading_style))
    elements.append(Spacer(1, 10))

    for line in important_points.split("\n"):
        line = line.strip()

        if not line:
            continue

        if line.startswith("-"):
            elements.append(Paragraph("• " + line[1:].strip(), normal_style))
        else:
            elements.append(Paragraph("• " + line, normal_style))

        elements.append(Spacer(1, 8))

    # Glossary
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Glossary", heading_style))
    elements.append(Spacer(1, 10))
    for term, meaning in glossary:
        elements.append(Paragraph(f"• <b>{term}</b>: {meaning}", normal_style))
        elements.append(Spacer(1, 6))

    # Timestamped notes
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Timestamped Notes", heading_style))
    elements.append(Spacer(1, 10))
    for line in timestamped_notes:
        content = line[2:].strip() if line.startswith("- ") else line
        elements.append(Paragraph("• " + content, normal_style))
        elements.append(Spacer(1, 6))

    # MCQs
    elements.append(PageBreak())
    elements.append(Paragraph("Practice MCQs", heading_style))
    elements.append(Spacer(1, 10))
    for line in mcqs.split("\n"):
        line = line.strip()
        if line:
            elements.append(Paragraph(line, normal_style))
            elements.append(Spacer(1, 6))

    # Flashcards
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Flashcards", heading_style))
    elements.append(Spacer(1, 10))
    for line in flashcards.split("\n"):
        line = line.strip()
        if line:
            elements.append(Paragraph(line, normal_style))
            elements.append(Spacer(1, 6))

    doc.build(elements, onFirstPage=add_watermark, onLaterPages=add_watermark)
    return pdf_path

# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return "OK", 200


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("audio")
    language = request.form.get("language")

    if not file or file.filename == "":
        return "No file uploaded.", 400

    if language not in ["en", "ml", "hi"]:
        return "Invalid language selected.", 400

    filepath = os.path.join(UPLOAD_FOLDER, "uploaded_audio.mp3")
    file.save(filepath)

    print("\n===== TRANSCRIBING AUDIO =====\n")

    model = get_whisper_model()
    segments_raw, info = model.transcribe(filepath, beam_size=1)

    segments = []
    transcript = ""

    for seg in segments_raw:
        text = seg.text.strip()
        if text:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": text
            })
            transcript += text + " "

    if not transcript.strip():
        return "Transcription failed.", 500

    print("\n===== CLEANING TRANSCRIPT =====\n")
    cleaned = clean_transcript(transcript)

    print("Detecting definitions...")
    definitions = detect_definitions(cleaned)

    print("Generating topic-wise summaries...")
    segment_groups = chunk_segments_by_word_limit(segments, max_words=180)
    timestamped_chunks = []

    for group in segment_groups:
        chunk_text_value = combine_segments_to_text(group)
        cleaned_chunk = clean_transcript(chunk_text_value)
        timestamped_chunks.append({
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": cleaned_chunk
        })

    topic_summaries = summarize_topic_chunks(timestamped_chunks)
    timestamped_notes = build_timestamped_notes(topic_summaries)

    print("Generating overall summary...")
    summary = summarize_text(cleaned)
    summary = summary.replace("\n", " ")

    print("Generating topic title...")
    topic_title = generate_topic_title(summary)

    print("Generating important notes...")
    important_notes = generate_important_notes(summary)

    print("Merging important notes and definitions...")
    important_points = merge_important_notes_and_definitions(important_notes, definitions)

    print("Generating glossary...")
    glossary = generate_glossary(definitions, cleaned)

    print("Generating MCQs...")
    mcq_source = cleaned if len(cleaned.split()) < 1200 else " ".join(cleaned.split()[:1200])
    mcqs = generate_mcqs(mcq_source)

    print("Generating flashcards...")
    flashcard_source = important_points if len(important_points.split()) < 900 else " ".join(important_points.split()[:900])
    flashcards = generate_flashcards(flashcard_source)

    print("Translating outputs if needed...")

    important_points = translate_bullet_text_block(important_points, language)
    timestamped_notes = translate_bullet_lines(timestamped_notes, language)
    glossary = translate_glossary(glossary, language)
    mcqs = translate_text_block(mcqs, language)
    flashcards = translate_text_block(flashcards, language)

    if language != "en":
        topic_title = GoogleTranslator(source="auto", target=language).translate(topic_title)

    print("Generating PDF...")
    pdf_path = generate_pdf(
        topic_title=topic_title,
        important_points=important_points,
        language=language,
        glossary=glossary,
        timestamped_notes=timestamped_notes,
        mcqs=mcqs,
        flashcards=flashcards
    )

    return send_file(pdf_path, as_attachment=True)

# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":
    print("Starting AI Lecture Notes Server...")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)