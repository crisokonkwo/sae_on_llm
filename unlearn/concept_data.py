"""Build a concept dataset: concept-positive (forget) and concept-negative
(retain) corpora, plus a probe set, all from a single YAML spec.

Sources
-------
* **Forget corpus** (concept-positive): Wikipedia article(s) named in the
  spec. Sentences are kept only if they contain at least one of the
  spec's positive terms (required or optional).
* **Retain corpus** (concept-negative): two modes
    - ``wikipedia_random`` (default): sample random Wikipedia titles, keep
      only sentences that contain none of the concept terms.
    - ``hf_dataset``: stream a chosen HF dataset (e.g. the same Pile slice
      the SAE was harvested on, with a ``skip_samples`` offset to stay
      outside the train/val/test slices).
* **Probes**: hand-written; declared inline in the YAML.

No external deps beyond ``pyyaml`` (which is already required for SAE
configs) and ``datasets`` for the HF-dataset retain source.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator


USER_AGENT = "sae-on-llm-research/0.1 (concept-data builder)"


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ConceptProbe:
    prompt: str
    paraphrases: list[str] = field(default_factory=list)
    expected_completion: str | None = None
    notes: str = ""

    def all_prompts(self) -> list[str]:
        """Returns [prompt, *paraphrases] de-duplicated, preserving order."""
        seen = set()
        out: list[str] = []
        for p in [self.prompt, *self.paraphrases]:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out


@dataclass
class ConceptDataSpec:
    name: str
    description: str = ""
    positive_required_terms: list[str] = field(default_factory=list)
    positive_optional_terms: list[str] = field(default_factory=list)
    wikipedia_titles: list[str] = field(default_factory=list)
    probes: list[ConceptProbe] = field(default_factory=list)
    # build settings
    target_forget_sentences: int = 1000
    target_retain_sentences: int = 1000
    retain_source: str = "wikipedia_random"      # or "hf_dataset"
    retain_hf_dataset: str | None = None
    retain_hf_config: str | None = None
    retain_hf_split: str = "train"
    retain_hf_text_field: str = "text"
    retain_skip_samples: int = 0
    min_sentence_chars: int = 30
    max_sentence_chars: int = 500
    lang: str = "en"

    @property
    def all_positive_terms(self) -> list[str]:
        return list(self.positive_required_terms) + list(self.positive_optional_terms)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConceptDataSpec":
        import yaml
        raw = yaml.safe_load(Path(path).read_text()) or {}
        probes_raw = raw.pop("probes", []) or []
        probes = [ConceptProbe(**p) for p in probes_raw]
        return cls(probes=probes, **raw)


@dataclass
class CorpusSentence:
    text: str
    source: str
    is_positive: bool
    matched_terms: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Wikipedia fetchers (stdlib only; uses the MediaWiki action API)
# ---------------------------------------------------------------------------
def _api_get(lang: str, params: dict, timeout: float = 30.0,
             retries: int = 2, backoff: float = 1.5) -> dict:
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            if attempt < retries:
                time.sleep(backoff ** attempt)
            else:
                raise
    raise RuntimeError(f"unreachable; last error was {last!r}")


def fetch_wikipedia_text(title: str, lang: str = "en", timeout: float = 30.0) -> str:
    """Fetch the plain-text extract for one Wikipedia article."""
    data = _api_get(lang, {
        "action": "query", "prop": "extracts", "explaintext": "true",
        "titles": title, "format": "json", "redirects": "true",
        "exsectionformat": "plain",
    }, timeout=timeout)
    pages = data.get("query", {}).get("pages", {}) or {}
    if not pages:
        return ""
    page = next(iter(pages.values()))
    return page.get("extract", "") or ""


def fetch_random_wikipedia_titles(n: int, lang: str = "en", timeout: float = 30.0) -> list[str]:
    """Sample ``n`` random main-namespace Wikipedia titles."""
    out: list[str] = []
    while len(out) < n:
        batch = min(n - len(out), 10)   # MediaWiki API caps rnlimit at 10 for anon
        data = _api_get(lang, {
            "action": "query", "list": "random", "rnnamespace": "0",
            "rnlimit": str(batch), "format": "json",
        }, timeout=timeout)
        for r in data.get("query", {}).get("random", []) or []:
            out.append(r["title"])
        if not data.get("query", {}).get("random"):
            # Defensive: avoid spinning forever if the API misbehaves.
            break
    return out[:n]


# ---------------------------------------------------------------------------
# Sentence splitting + term matching
# ---------------------------------------------------------------------------
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d\"'])")
# Lines starting with `==` are Wikipedia section headers in the plaintext
# extract format.
_SECTION_HEADER = re.compile(r"^\s*=+.*=+\s*$")


def split_sentences(text: str) -> list[str]:
    """Split English text into sentences. Strips Wikipedia section headers
    and blank lines first."""
    cleaned: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or _SECTION_HEADER.match(line):
            continue
        cleaned.append(line)
    joined = " ".join(cleaned)
    return [s.strip() for s in _SENTENCE_SPLIT.split(joined) if s.strip()]


def sentence_matches(sent: str, terms: Iterable[str], case_insensitive: bool = True) -> list[str]:
    """Return the subset of ``terms`` that appear in ``sent`` with word
    boundaries. Multi-word terms allow flexible whitespace inside."""
    flags = re.IGNORECASE if case_insensitive else 0
    out: list[str] = []
    for t in terms:
        if not t:
            continue
        parts = [re.escape(w) for w in t.split()]
        body = r"\s+".join(parts)
        pattern = rf"\b{body}\b"
        if re.search(pattern, sent, flags):
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------
TextFetcher = Callable[[str], str]
TitleSampler = Callable[[int], list[str]]


def build_forget_corpus(
    spec: ConceptDataSpec,
    *,
    fetch_text: TextFetcher | None = None,
    progress: bool = True,
) -> list[CorpusSentence]:
    """Build the concept-positive sentence corpus from the spec's Wikipedia
    article list. Sentences are kept only if they mention at least one of
    the positive terms."""
    fetch = fetch_text or (lambda title: fetch_wikipedia_text(title, lang=spec.lang))
    all_terms = spec.all_positive_terms
    if not all_terms:
        raise ValueError("spec has no positive terms; set positive_required_terms")
    if not spec.wikipedia_titles:
        raise ValueError("spec has no wikipedia_titles; set at least one source article")

    sentences: list[CorpusSentence] = []
    seen: set[str] = set()
    iterable: Iterable[str] = spec.wikipedia_titles
    if progress:
        try:
            from tqdm.auto import tqdm
            iterable = tqdm(iterable, desc="[forget]", unit="article")
        except ImportError:
            pass
    for title in iterable:
        try:
            text = fetch(title)
        except Exception as e:
            print(f"[concept_data] failed to fetch {title!r}: {e}", file=sys.stderr)
            continue
        if not text:
            print(f"[concept_data] empty article: {title!r}", file=sys.stderr)
            continue
        for sent in split_sentences(text):
            if not (spec.min_sentence_chars <= len(sent) <= spec.max_sentence_chars):
                continue
            if sent in seen:
                continue
            matched = sentence_matches(sent, all_terms)
            if not matched:
                continue
            seen.add(sent)
            sentences.append(CorpusSentence(
                text=sent, source=f"wikipedia:{title}",
                is_positive=True, matched_terms=matched,
            ))
            if len(sentences) >= spec.target_forget_sentences:
                return sentences
    return sentences


def _build_retain_from_wikipedia_random(
    spec: ConceptDataSpec,
    *,
    fetch_text: TextFetcher,
    sample_titles: TitleSampler,
    progress: bool,
    max_articles: int = 5000,
) -> list[CorpusSentence]:
    sentences: list[CorpusSentence] = []
    seen: set[str] = set()
    all_terms = spec.all_positive_terms
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=spec.target_retain_sentences, desc="[retain]", unit="sentence")
        except ImportError:
            bar = None

    articles_seen = 0
    while len(sentences) < spec.target_retain_sentences and articles_seen < max_articles:
        try:
            titles = sample_titles(min(20, max_articles - articles_seen))
        except Exception as e:
            print(f"[concept_data] random title fetch failed: {e}", file=sys.stderr)
            break
        if not titles:
            break
        for title in titles:
            articles_seen += 1
            try:
                text = fetch_text(title)
            except Exception:
                continue
            if not text:
                continue
            for sent in split_sentences(text):
                if not (spec.min_sentence_chars <= len(sent) <= spec.max_sentence_chars):
                    continue
                if sent in seen:
                    continue
                if sentence_matches(sent, all_terms):
                    # Skip sentences that mention the concept (rare here, but possible).
                    continue
                seen.add(sent)
                sentences.append(CorpusSentence(
                    text=sent, source=f"wikipedia_random:{title}",
                    is_positive=False,
                ))
                if bar is not None:
                    bar.update(1)
                if len(sentences) >= spec.target_retain_sentences:
                    break
            if len(sentences) >= spec.target_retain_sentences:
                break
    if bar is not None:
        bar.close()
    return sentences


def _build_retain_from_hf(
    spec: ConceptDataSpec,
    *,
    progress: bool,
    max_docs: int = 50_000,
) -> list[CorpusSentence]:
    from datasets import load_dataset

    if not spec.retain_hf_dataset:
        raise ValueError("retain_source='hf_dataset' requires retain_hf_dataset to be set")
    ds = load_dataset(
        spec.retain_hf_dataset, spec.retain_hf_config,
        split=spec.retain_hf_split, streaming=True,
    )
    if spec.retain_skip_samples:
        ds = ds.skip(spec.retain_skip_samples)
    ds = ds.take(max_docs)

    sentences: list[CorpusSentence] = []
    seen: set[str] = set()
    all_terms = spec.all_positive_terms
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=spec.target_retain_sentences, desc="[retain]", unit="sentence")
        except ImportError:
            bar = None

    for ex in ds:
        text = ex.get(spec.retain_hf_text_field, "")
        if not text:
            continue
        for sent in split_sentences(text):
            if not (spec.min_sentence_chars <= len(sent) <= spec.max_sentence_chars):
                continue
            if sent in seen:
                continue
            if sentence_matches(sent, all_terms):
                continue
            seen.add(sent)
            sentences.append(CorpusSentence(
                text=sent,
                source=f"hf:{spec.retain_hf_dataset}",
                is_positive=False,
            ))
            if bar is not None:
                bar.update(1)
            if len(sentences) >= spec.target_retain_sentences:
                break
        if len(sentences) >= spec.target_retain_sentences:
            break
    if bar is not None:
        bar.close()
    return sentences


def build_retain_corpus(
    spec: ConceptDataSpec,
    *,
    fetch_text: TextFetcher | None = None,
    sample_titles: TitleSampler | None = None,
    progress: bool = True,
) -> list[CorpusSentence]:
    """Build the concept-negative sentence corpus.

    Modes (selected by ``spec.retain_source``):
      * ``wikipedia_random`` (default): sample random Wikipedia articles and
        keep sentences that do not mention any concept term.
      * ``hf_dataset``: stream from a chosen HF dataset (e.g. the Pile slice
        the SAE was harvested on, with ``retain_skip_samples`` to stay
        outside the train/val/test ranges).
    """
    if spec.retain_source == "wikipedia_random":
        return _build_retain_from_wikipedia_random(
            spec,
            fetch_text=fetch_text or (lambda t: fetch_wikipedia_text(t, lang=spec.lang)),
            sample_titles=sample_titles or (lambda n: fetch_random_wikipedia_titles(n, lang=spec.lang)),
            progress=progress,
        )
    if spec.retain_source == "hf_dataset":
        return _build_retain_from_hf(spec, progress=progress)
    raise ValueError(f"Unknown retain_source: {spec.retain_source!r}")


# ---------------------------------------------------------------------------
# Statistics / inspection
# ---------------------------------------------------------------------------
@dataclass
class CorpusStatistics:
    n_sentences: int
    n_chars_mean: float
    n_chars_median: float
    n_chars_min: int
    n_chars_max: int
    n_words_mean: float
    n_words_median: float
    n_words_min: int
    n_words_max: int
    n_tokens_mean: float | None = None
    n_tokens_median: float | None = None
    n_tokens_total: int | None = None
    sources_top: list[tuple[str, int]] = field(default_factory=list)
    term_match_counts: dict[str, int] = field(default_factory=dict)


def _median(xs: list[int]) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    n = len(xs)
    if n % 2 == 1:
        return float(xs[n // 2])
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def corpus_statistics(
    sentences: list[CorpusSentence],
    *,
    tokenizer=None,
    top_sources: int = 5,
    term_counts_for: Iterable[str] | None = None,
) -> CorpusStatistics:
    from collections import Counter

    chars = [len(s.text) for s in sentences]
    words = [len(s.text.split()) for s in sentences]
    n_tokens_mean = n_tokens_median = n_tokens_total = None
    if tokenizer is not None and sentences:
        tok_counts = [len(tokenizer.encode(s.text, add_special_tokens=False)) for s in sentences]
        n_tokens_mean = sum(tok_counts) / len(tok_counts)
        n_tokens_median = _median(tok_counts)
        n_tokens_total = sum(tok_counts)
    source_counts = Counter(s.source for s in sentences)
    sources_top = source_counts.most_common(top_sources)
    term_match_counts: dict[str, int] = {}
    if term_counts_for is not None:
        for t in term_counts_for:
            term_match_counts[t] = sum(1 for s in sentences if t in s.matched_terms)
    return CorpusStatistics(
        n_sentences=len(sentences),
        n_chars_mean=sum(chars) / len(chars) if chars else 0.0,
        n_chars_median=_median(chars),
        n_chars_min=min(chars) if chars else 0,
        n_chars_max=max(chars) if chars else 0,
        n_words_mean=sum(words) / len(words) if words else 0.0,
        n_words_median=_median(words),
        n_words_min=min(words) if words else 0,
        n_words_max=max(words) if words else 0,
        n_tokens_mean=n_tokens_mean,
        n_tokens_median=n_tokens_median,
        n_tokens_total=n_tokens_total,
        sources_top=sources_top,
        term_match_counts=term_match_counts,
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def write_corpus(sentences: list[CorpusSentence], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in sentences:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")


def read_corpus(path: str | Path) -> list[CorpusSentence]:
    out: list[CorpusSentence] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(CorpusSentence(**d))
    return out


def write_probes(probes: list[ConceptProbe], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in probes:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def read_probes(path: str | Path) -> list[ConceptProbe]:
    out: list[ConceptProbe] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(ConceptProbe(**json.loads(line)))
    return out


def write_spec(spec: ConceptDataSpec, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(spec), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Markdown inspection renderer
# ---------------------------------------------------------------------------
def render_inspection_md(
    spec: ConceptDataSpec,
    forget: list[CorpusSentence],
    retain: list[CorpusSentence],
    *,
    forget_stats: CorpusStatistics,
    retain_stats: CorpusStatistics,
    examples_per_corpus: int = 8,
) -> str:
    lines: list[str] = []
    lines.append(f"# Concept dataset: {spec.name}")
    if spec.description:
        lines.append("")
        lines.append(spec.description)
    lines.append("")
    lines.append("## Spec")
    lines.append(f"- required terms: {spec.positive_required_terms}")
    lines.append(f"- optional terms: {spec.positive_optional_terms}")
    lines.append(f"- wikipedia sources: {spec.wikipedia_titles}")
    lines.append(f"- retain source: {spec.retain_source}")
    if spec.retain_source == "hf_dataset":
        lines.append(f"  - hf_dataset: {spec.retain_hf_dataset} / {spec.retain_hf_config}")
        lines.append(f"  - skip_samples: {spec.retain_skip_samples}")
    lines.append(f"- sentence length window: "
                 f"[{spec.min_sentence_chars}, {spec.max_sentence_chars}] chars")
    lines.append("")

    def _stats_block(name: str, st: CorpusStatistics) -> None:
        lines.append(f"## {name}")
        lines.append(f"- sentences: **{st.n_sentences}**")
        lines.append(f"- chars: mean {st.n_chars_mean:.1f}, median {st.n_chars_median:.0f}, "
                     f"min {st.n_chars_min}, max {st.n_chars_max}")
        lines.append(f"- words: mean {st.n_words_mean:.1f}, median {st.n_words_median:.0f}, "
                     f"min {st.n_words_min}, max {st.n_words_max}")
        if st.n_tokens_mean is not None:
            lines.append(f"- tokens: mean {st.n_tokens_mean:.1f}, median {st.n_tokens_median:.0f}, "
                         f"total {st.n_tokens_total:,}")
        if st.sources_top:
            lines.append(f"- top sources:")
            for src, n in st.sources_top:
                lines.append(f"    - {src}: {n}")
        if st.term_match_counts:
            lines.append("- term-match counts:")
            for term, n in sorted(st.term_match_counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"    - `{term}`: {n}")
        lines.append("")

    _stats_block("Forget corpus (concept-positive)", forget_stats)
    _stats_block("Retain corpus (concept-negative)", retain_stats)

    lines.append("## Example forget sentences")
    for s in forget[:examples_per_corpus]:
        lines.append(f"- _{s.source}_  (terms: {s.matched_terms})")
        lines.append(f"  > {s.text}")
    lines.append("")
    lines.append("## Example retain sentences")
    for s in retain[:examples_per_corpus]:
        lines.append(f"- _{s.source}_")
        lines.append(f"  > {s.text}")
    lines.append("")

    lines.append(f"## Probes ({len(spec.probes)})")
    for p in spec.probes:
        lines.append(f"- **{p.prompt}**")
        if p.expected_completion is not None:
            lines.append(f"  - expected completion: `{p.expected_completion}`")
        for para in p.paraphrases:
            lines.append(f"  - paraphrase: {para}")
        if p.notes:
            lines.append(f"  - notes: {p.notes}")
    return "\n".join(lines)
