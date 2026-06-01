"""Estimate how well a candidate concept is represented in the harvest
corpus used (or about to be used) for SAE training.

A *concept* is a list of terms that must co-occur within a configurable
window. The four shapes the user typically wants are all reducible to this:

* single  ``"internet"``                       -> ``["internet"]``
* pair    ``("computer", "internet")``         -> ``["computer", "internet"]``
* triple  ``("computer", "uses", "internet")`` -> ``["computer", "uses", "internet"]``
* list of any of the above                     -> one report per item

Optional structure: each term can be marked *required* or *optional*. SRO
triples often want only ``(subject, object)`` required and the relation
optional, because the relation is usually encoded distributionally rather
than as a literal token.

What is measured (per concept, on a streamed corpus):

* per-term occurrence count and document frequency,
* documents containing every required term anywhere,
* documents containing every required term within a sliding window of
  ``window_words`` whitespace tokens,
* a few example windows so the matching can be eyeballed,
* a coarse verdict ("strong" / "moderate" / "weak") based on co-occurrence
  density.

This file has no Torch / Transformers dependency and runs on CPU in
seconds for ~10k documents.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


# ---------------------------------------------------------------------------
# Concept specification
# ---------------------------------------------------------------------------
@dataclass
class Term:
    text: str
    required: bool = True

    def __post_init__(self) -> None:
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("Empty term")


@dataclass
class ConceptSpec:
    """A concept the SAE-training corpus should sufficiently represent.

    ``kind`` is informational only ("single" / "pair" / "triple" / "custom").
    The actual semantics is the term list + the ``required`` flags.
    """
    name: str
    terms: list[Term]
    kind: str = "custom"

    @property
    def required_terms(self) -> list[Term]:
        return [t for t in self.terms if t.required]

    @property
    def all_term_texts(self) -> list[str]:
        return [t.text for t in self.terms]


# ---------------------------------------------------------------------------
# Concept parsing (user-friendly inputs -> ConceptSpec)
# ---------------------------------------------------------------------------
def _kind_from_n(n: int) -> str:
    return {1: "single", 2: "pair", 3: "triple"}.get(n, "custom")


def parse_concept(spec: Any, name: str | None = None) -> ConceptSpec:
    """Coerce user-friendly inputs into a :class:`ConceptSpec`.

    Accepted forms::

        "internet"                                  -> single, 1 required term
        ["computer", "internet"]                    -> pair, both required
        ("computer", "uses", "internet")            -> triple, all required
        {"name": ..., "terms": ["A","B"]}           -> explicit, all required
        {"name": ..., "required": [...], "optional": [...]}  -> mixed

    For tuples/lists of length 3, the *middle* element is taken to be a
    relation and marked optional by default (override with the dict form).
    """
    if isinstance(spec, str):
        return ConceptSpec(
            name=name or spec,
            terms=[Term(spec, required=True)],
            kind="single",
        )

    if isinstance(spec, (list, tuple)):
        items = list(spec)
        if len(items) == 0:
            raise ValueError("Empty concept term list")
        terms: list[Term]
        if len(items) == 3:
            # Treat as (subject, relation, object) by default.
            s, r, o = items
            terms = [
                Term(str(s), required=True),
                Term(str(r), required=False),
                Term(str(o), required=True),
            ]
        else:
            terms = [Term(str(t), required=True) for t in items]
        return ConceptSpec(
            name=name or " + ".join(map(str, items)),
            terms=terms,
            kind=_kind_from_n(len(items)),
        )

    if isinstance(spec, dict):
        req = list(spec.get("required", []))
        opt = list(spec.get("optional", []))
        flat = list(spec.get("terms", []))
        if flat and not (req or opt):
            req = flat
        if not (req or opt):
            raise ValueError(f"Concept dict missing terms: {spec}")
        terms = [Term(t, required=True) for t in req] + [Term(t, required=False) for t in opt]
        return ConceptSpec(
            name=name or spec.get("name") or " + ".join(req + opt),
            terms=terms,
            kind=spec.get("kind", "custom"),
        )

    raise TypeError(f"Cannot parse concept spec: {spec!r}")


def parse_cli_concept_string(s: str, name: str | None = None) -> ConceptSpec:
    """Parse a CLI-friendly string. Term separator: '|' (or ':' as legacy).

    A term may be suffixed ``?`` to mark it optional::

        "Eiffel Tower|Paris"            pair, both required
        "computer|uses?|internet"       SRO with relation optional
        "Eiffel Tower"                  single
    """
    sep = "|" if "|" in s else ":"
    raw = [t for t in (p.strip() for p in s.split(sep)) if t]
    if not raw:
        raise ValueError(f"Empty concept string: {s!r}")
    if len(raw) == 1:
        return parse_concept(raw[0], name=name)
    terms: list[Term] = []
    for t in raw:
        if t.endswith("?"):
            terms.append(Term(t[:-1], required=False))
        else:
            terms.append(Term(t, required=True))
    return ConceptSpec(
        name=name or " + ".join(t.text for t in terms),
        terms=terms,
        kind=_kind_from_n(len(raw)),
    )


# ---------------------------------------------------------------------------
# Matching utilities
# ---------------------------------------------------------------------------
def _term_regex(term: str, word_boundary: bool, case_insensitive: bool) -> re.Pattern:
    """Compile a regex that matches a term as a multi-word phrase, optionally
    with word-boundary anchors. Whitespace inside the term is allowed to be
    any run of whitespace in the text.
    """
    parts = [re.escape(w) for w in term.split()]
    body = r"\s+".join(parts)
    if word_boundary:
        body = rf"\b{body}\b"
    flags = re.IGNORECASE if case_insensitive else 0
    return re.compile(body, flags)


def _word_positions(text: str, pattern: re.Pattern) -> list[int]:
    """Return the whitespace-token indices at which the pattern starts a match.

    We tokenise the text once into words with character offsets, then check
    which word starts contain a match. This gives a consistent notion of
    "co-occurrence within W words" regardless of phrase length.
    """
    words = []      # list of (word_start_char, word_end_char)
    for m in re.finditer(r"\S+", text):
        words.append((m.start(), m.end()))
    if not words:
        return []
    positions: list[int] = []
    for m in pattern.finditer(text):
        cs = m.start()
        # Binary search for the word containing char offset cs.
        lo, hi = 0, len(words) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            ws, we = words[mid]
            if cs < ws:
                hi = mid - 1
            elif cs >= we:
                lo = mid + 1
            else:
                positions.append(mid)
                break
        else:
            # Match starts between words (rare with our pattern); skip.
            pass
    return positions


def _has_window_cooccurrence(
    term_positions: list[list[int]], window_words: int
) -> tuple[bool, list[int] | None]:
    """Given per-term sorted word-index lists, check whether there exists a
    window of ``window_words`` whitespace tokens containing at least one
    occurrence of every term.

    Returns ``(found, sample_positions)`` where ``sample_positions`` is one
    valid combination of positions (one per term) inside such a window, used
    for context extraction.
    """
    if any(len(p) == 0 for p in term_positions):
        return False, None
    # Merge into a stream of (word_idx, term_id), sorted by word_idx.
    stream: list[tuple[int, int]] = []
    for tid, plist in enumerate(term_positions):
        for p in plist:
            stream.append((p, tid))
    stream.sort()
    # Sliding window over word indices.
    n_terms = len(term_positions)
    counts = [0] * n_terms
    distinct = 0
    last_seen: list[int | None] = [None] * n_terms
    left = 0
    for right in range(len(stream)):
        p_r, t_r = stream[right]
        if counts[t_r] == 0:
            distinct += 1
        counts[t_r] += 1
        last_seen[t_r] = p_r
        # Shrink window from left while it exceeds the size.
        while p_r - stream[left][0] >= window_words:
            p_l, t_l = stream[left]
            counts[t_l] -= 1
            if counts[t_l] == 0:
                distinct -= 1
                # Update last_seen for that term (find any remaining position).
                # Cheap rescan only if needed by callers.
            left += 1
        if distinct == n_terms:
            # Find one representative position per term inside [left..right].
            rep: list[int | None] = [None] * n_terms
            for i in range(left, right + 1):
                p_i, t_i = stream[i]
                if rep[t_i] is None:
                    rep[t_i] = p_i
            return True, [p for p in rep if p is not None]
    return False, None


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
@dataclass
class CoverageReport:
    concept_name: str
    kind: str
    terms: list[dict]                 # serialized Term list with required flag
    n_docs_scanned: int = 0
    n_words_scanned: int = 0
    term_token_freq: dict[str, int] = field(default_factory=dict)
    term_doc_freq: dict[str, int] = field(default_factory=dict)
    docs_with_any_required: int = 0
    docs_with_all_required: int = 0
    docs_with_window_cooccurrence: int = 0
    window_words: int = 64
    example_contexts: list[dict] = field(default_factory=list)
    verdict: str = "unknown"
    verdict_notes: list[str] = field(default_factory=list)
    # SAE-feature-allocation heuristic (filled by _assign_verdict).
    cooccurrence_per_million_tokens: float = 0.0
    sae_feature_likelihood: str = "unknown"

    @property
    def density_per_doc(self) -> float:
        if self.n_docs_scanned == 0:
            return 0.0
        return self.docs_with_window_cooccurrence / self.n_docs_scanned

    def to_dict(self) -> dict:
        d = asdict(self)
        d["density_per_doc"] = self.density_per_doc
        return d


def _assign_verdict(
    report: CoverageReport,
    min_per_term_tokens: int = 50,
    *,
    words_to_tokens_ratio: float = 1.3,
    likely_threshold_per_million: float = 10.0,
    marginal_threshold_per_million: float = 1.0,
) -> None:
    """Coarse traffic-light verdict + SAE-feature-allocation heuristic.

    Two perspectives:
      * Doc-level traffic light: strong/moderate/weak by window co-occurrence
        density across documents.
      * SAE-level likelihood: likely/marginal/unlikely based on co-occurrence
        events per million estimated tokens. This is the metric most
        predictive of whether the SAE will allocate a clean dictionary slot.
    """
    notes: list[str] = []
    required_terms = [t for t in report.terms if t["required"]]

    # Estimate co-occurrence events per million tokens (the SAE-relevant scale).
    n_tokens_est = max(report.n_words_scanned * words_to_tokens_ratio, 1.0)
    per_million = report.docs_with_window_cooccurrence / n_tokens_est * 1e6
    report.cooccurrence_per_million_tokens = per_million
    if per_million >= likely_threshold_per_million:
        report.sae_feature_likelihood = "likely"
    elif per_million >= marginal_threshold_per_million:
        report.sae_feature_likelihood = "marginal"
    else:
        report.sae_feature_likelihood = "unlikely"

    too_rare = [t["text"] for t in required_terms
                if report.term_token_freq.get(t["text"], 0) < min_per_term_tokens]
    if too_rare:
        report.verdict = "weak"
        notes.append(
            f"required terms with < {min_per_term_tokens} occurrences: {too_rare}"
        )
        notes.append(
            f"SAE feature likelihood: {report.sae_feature_likelihood} "
            f"({per_million:.2f} events / million tokens)"
        )
        report.verdict_notes = notes
        return

    d = report.density_per_doc
    if d >= 1e-2:
        report.verdict = "strong"
    elif d >= 1e-3:
        report.verdict = "moderate"
    else:
        report.verdict = "weak"
    notes.append(f"window co-occurrence density: {d:.4g} per doc")
    notes.append(
        f"SAE feature likelihood: {report.sae_feature_likelihood} "
        f"({per_million:.2f} events / million tokens; "
        f"likely >= {likely_threshold_per_million}, marginal >= {marginal_threshold_per_million})"
    )
    if report.docs_with_window_cooccurrence == 0 and report.docs_with_all_required > 0:
        notes.append(
            "terms co-occur in same doc but not within window; consider --window-words"
        )
    report.verdict_notes = notes


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------
def scan_concepts(
    text_iter: Iterable[str],
    concepts: Sequence[ConceptSpec],
    *,
    window_words: int = 64,
    case_insensitive: bool = True,
    word_boundary: bool = True,
    max_examples: int = 5,
    context_words: int = 24,
    n_docs_limit: int | None = None,
    n_tokens_actual: int | None = None,
) -> list[CoverageReport]:
    """Stream ``text_iter`` once and collect coverage stats for every
    concept simultaneously. Returns one :class:`CoverageReport` per concept.
    """
    # Precompile patterns per term (deduplicate identical terms across concepts).
    unique_terms: dict[str, re.Pattern] = {}
    for c in concepts:
        for t in c.terms:
            if t.text not in unique_terms:
                unique_terms[t.text] = _term_regex(t.text, word_boundary, case_insensitive)

    reports: list[CoverageReport] = []
    for c in concepts:
        rep = CoverageReport(
            concept_name=c.name,
            kind=c.kind,
            terms=[{"text": t.text, "required": t.required} for t in c.terms],
            window_words=window_words,
        )
        for t in c.terms:
            rep.term_token_freq.setdefault(t.text, 0)
            rep.term_doc_freq.setdefault(t.text, 0)
        reports.append(rep)

    n_docs = 0
    for doc_idx, text in enumerate(text_iter):
        if n_docs_limit is not None and n_docs >= n_docs_limit:
            break
        n_docs += 1
        # Approximate token count via whitespace word count.
        word_count = len(text.split())

        # Per-term positions cached for this doc (only computed if any concept needs them).
        per_term_positions: dict[str, list[int]] = {}

        for rep, c in zip(reports, concepts):
            # Doc-level term stats first.
            doc_has_term: dict[str, bool] = {}
            for t in c.terms:
                if t.text not in per_term_positions:
                    per_term_positions[t.text] = _word_positions(text, unique_terms[t.text])
                positions = per_term_positions[t.text]
                if positions:
                    doc_has_term[t.text] = True
                    rep.term_token_freq[t.text] += len(positions)
                    rep.term_doc_freq[t.text] += 1
                else:
                    doc_has_term[t.text] = False

            required = [t.text for t in c.terms if t.required]
            any_required = any(doc_has_term[t] for t in required) if required else False
            all_required = all(doc_has_term[t] for t in required) if required else False
            if any_required:
                rep.docs_with_any_required += 1
            if all_required:
                rep.docs_with_all_required += 1
                # Window co-occurrence only if all required are present at all.
                term_positions = [per_term_positions[t] for t in required]
                ok, rep_positions = _has_window_cooccurrence(term_positions, window_words)
                if ok:
                    rep.docs_with_window_cooccurrence += 1
                    if len(rep.example_contexts) < max_examples and rep_positions is not None:
                        # Build a context snippet around the window's center.
                        center = sum(rep_positions) // len(rep_positions)
                        words = text.split()
                        lo = max(0, center - context_words)
                        hi = min(len(words), center + context_words + 1)
                        snippet = " ".join(words[lo:hi])
                        rep.example_contexts.append({
                            "doc_idx": doc_idx,
                            "word_center": center,
                            "context": snippet,
                            "required_terms": required,
                        })

        for rep in reports:
            rep.n_words_scanned += word_count

    for rep in reports:
        rep.n_docs_scanned = n_docs
        if n_tokens_actual is not None and n_tokens_actual > 0:
            # Override word-based estimate with the real harvest token count.
            # _assign_verdict uses report.n_words_scanned * ratio internally,
            # so we set ratio such that n_words * ratio = n_tokens_actual.
            ratio = n_tokens_actual / max(rep.n_words_scanned, 1)
            _assign_verdict(rep, words_to_tokens_ratio=ratio)
        else:
            _assign_verdict(rep)
    return reports


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def coverage_check_from_dataset(
    concepts: Sequence[ConceptSpec],
    *,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    text_field: str,
    n_docs: int,
    skip_samples: int = 0,
    window_words: int = 64,
    case_insensitive: bool = True,
    word_boundary: bool = True,
    max_examples: int = 5,
    context_words: int = 24,
    progress: bool = True,
    progress_desc: str = "[coverage]",
    n_tokens_actual: int | None = None,
) -> list[CoverageReport]:
    """Stream an HF dataset (matching the harvest pipeline) and run
    :func:`scan_concepts`. Imports ``datasets`` lazily.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True)
    if skip_samples:
        ds = ds.skip(skip_samples)
    ds = ds.take(n_docs)

    def _it() -> Iterator[str]:
        if progress:
            try:
                from tqdm.auto import tqdm
                bar = tqdm(total=n_docs, desc=progress_desc, unit="doc", smoothing=0.05)
            except ImportError:
                bar = None
        else:
            bar = None
        for ex in ds:
            text = ex.get(text_field)
            if text:
                yield text
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()

    return scan_concepts(
        _it(),
        concepts,
        n_tokens_actual=n_tokens_actual,
        window_words=window_words,
        case_insensitive=case_insensitive,
        word_boundary=word_boundary,
        max_examples=max_examples,
        context_words=context_words,
        n_docs_limit=n_docs,
    )


def write_reports(reports: Sequence[CoverageReport], path: str) -> None:
    payload = {r.concept_name: r.to_dict() for r in reports}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Harvested-shard-aware entry points
# ---------------------------------------------------------------------------
REQUIRED_META_FIELDS = ("dataset_name",)


def _load_shard_meta(shard_dir: "Path | str") -> dict:
    """Read ``meta.json`` from a shard directory and validate the fields
    needed to re-stream the original text slice."""
    from pathlib import Path as _Path
    meta_path = _Path(shard_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No meta.json under {shard_dir}. Was this dir created by harvest_activations?"
        )
    with open(meta_path) as f:
        meta = json.load(f)
    missing = [k for k in REQUIRED_META_FIELDS if not meta.get(k)]
    if missing:
        raise ValueError(
            f"meta.json in {shard_dir} is missing required field(s): {missing}"
        )
    return meta


def _shard_dir_n_docs(meta: dict) -> int:
    """Best estimate of how many documents this shard set was harvested from.

    Prefer ``docs_processed`` (actual count after the run finished); fall back
    to ``max_samples`` from the harvest config.
    """
    n = meta.get("docs_processed")
    if isinstance(n, int) and n > 0:
        return n
    n = meta.get("max_samples")
    if isinstance(n, int) and n > 0:
        return n
    raise ValueError(
        "Cannot determine docs to scan: meta.json has neither docs_processed "
        "nor a positive max_samples. Pass --n-docs explicitly."
    )


def coverage_check_from_shard_dir(
    shard_dir: "Path | str",
    concepts: Sequence[ConceptSpec],
    *,
    n_docs_override: int | None = None,
    split_label: str | None = None,
    window_words: int = 64,
    case_insensitive: bool = True,
    word_boundary: bool = True,
    max_examples: int = 5,
    context_words: int = 24,
    progress: bool = True,
    use_harvest_token_count: bool = True,
) -> list[CoverageReport]:
    """Scan exactly the text slice that produced the activations in
    ``shard_dir``. Reads ``meta.json`` to recover
    ``dataset_name / config / split / text_field / skip_samples / max_samples``
    so the stream is reproducible.
    """
    meta = _load_shard_meta(shard_dir)
    n_docs = n_docs_override if n_docs_override is not None else _shard_dir_n_docs(meta)
    label = split_label or str(shard_dir)
    n_tokens_actual = None
    if use_harvest_token_count:
        tw = meta.get("tokens_written")
        if isinstance(tw, int) and tw > 0:
            n_tokens_actual = tw
    return coverage_check_from_dataset(
        concepts,
        dataset_name=meta["dataset_name"],
        dataset_config=meta.get("dataset_config"),
        dataset_split=meta.get("dataset_split", "train"),
        text_field=meta.get("text_field", "text"),
        n_docs=n_docs,
        skip_samples=int(meta.get("skip_samples", 0) or 0),
        window_words=window_words,
        case_insensitive=case_insensitive,
        word_boundary=word_boundary,
        max_examples=max_examples,
        context_words=context_words,
        progress=progress,
        progress_desc=f"[coverage:{label}]",
        n_tokens_actual=n_tokens_actual,
    )


def _is_splits_root(path: "Path | str") -> bool:
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.is_dir():
        return False
    return any((p / split).is_dir() and (p / split / "meta.json").exists()
               for split in ("train", "val", "test"))


def coverage_check_from_splits_root(
    root: "Path | str",
    concepts: Sequence[ConceptSpec],
    *,
    splits: Sequence[str] = ("train", "val", "test"),
    **kwargs,
) -> dict[str, list[CoverageReport]]:
    """Run :func:`coverage_check_from_shard_dir` on each present split under
    ``root``. Returns ``{split: [reports]}``.
    """
    from pathlib import Path as _Path
    root = _Path(root)
    out: dict[str, list[CoverageReport]] = {}
    for split in splits:
        sub = root / split
        if not (sub / "meta.json").exists():
            continue
        out[split] = coverage_check_from_shard_dir(
            sub, concepts, split_label=split, **kwargs,
        )
    if not out:
        raise FileNotFoundError(
            f"No train/val/test subdirs with meta.json found under {root}"
        )
    return out


def write_split_reports(
    reports_by_split: dict[str, Sequence[CoverageReport]], path: str,
) -> None:
    payload = {
        split: {r.concept_name: r.to_dict() for r in reports}
        for split, reports in reports_by_split.items()
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Discovery mode: propose candidate concepts from the corpus itself
# ---------------------------------------------------------------------------
_DISCOVERY_DETERMINER_STOPWORDS = {
    "The", "A", "An", "This", "That", "These", "Those",
    "His", "Her", "My", "Our", "Your", "Their", "Its",
    "Some", "Any", "Every", "All", "Many", "Most",
}
# Common leading words that produce noisy phrases (filtered after stripping determiners).
_DISCOVERY_GENERIC_LEADS = {
    "He", "She", "It", "We", "You", "They", "I",
    "When", "Where", "What", "Who", "Why", "How",
    "Today", "Yesterday", "Tomorrow", "Now", "Then",
    "Yes", "No", "But", "And", "Or", "So",
    "Mr", "Mrs", "Ms", "Dr",   # often noisy without a name following
}
# Tokens we never want anywhere in a candidate phrase.
_DISCOVERY_PHRASE_BLOCKLIST = set()  # extension point; empty default.

_CAPITALISED_TOKEN_RE = re.compile(r"\b([A-Z][A-Za-z'\-]+|[A-Z]{2,})\b")
_WORD_RE = re.compile(r"\S+")


@dataclass
class DiscoveredConcept:
    text: str
    n_words: int
    token_freq: int           # total occurrences across the scanned corpus
    doc_freq: int             # number of docs containing this phrase
    density: float            # doc_freq / n_docs_scanned
    score: float              # ranking score (see discover_concepts)
    # SAE-feature-allocation heuristic:
    #   how many *concept activation events* will the SAE see per million
    #   training tokens? Empirically, concepts firing > ~10 events/M get clean
    #   features; < 1/M usually get absorbed into broader features.
    per_million_tokens: float = 0.0
    sae_feature_likelihood: str = "unknown"  # "likely" | "marginal" | "unlikely"
    suggested_partners: list[dict] = field(default_factory=list)  # [{text, cooccurrence_count}]
    # Each entry: {"doc_idx": int, "word_center": int, "context": str}
    example_contexts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_concept_spec(self, with_partner: bool = True) -> "ConceptSpec":
        """Convert into a ConceptSpec usable by the regular coverage scan."""
        terms = [Term(self.text, required=True)]
        if with_partner and self.suggested_partners:
            terms.append(Term(self.suggested_partners[0]["text"], required=True))
        name = self.text
        if len(terms) > 1:
            name = f"{self.text} + {terms[1].text}"
        return ConceptSpec(name=name, terms=terms, kind=_kind_from_n(len(terms)))


def _extract_capitalised_phrases(
    text: str, ngram_max: int,
) -> tuple[list[tuple[str, tuple[int, int]]], list[tuple[int, int]]]:
    """Return (phrases, word_spans). ``phrases`` is a list of
    ``(phrase_text, (start_word_idx, end_word_idx_exclusive))`` for every
    capitalised n-gram of length 2..ngram_max found in ``text``.
    ``word_spans`` is the list of whitespace-token char spans (for ctx).
    """
    word_spans = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    words = [text[s:e] for s, e in word_spans]
    if not words:
        return [], []

    # Map each word index to whether the bare token is "capitalised" in our sense.
    def _is_cap_word(w: str) -> bool:
        # Strip leading punctuation that often hugs words (e.g. '(' or '"').
        ws = w.lstrip("\"'(\u201c\u2018[{")
        ws = ws.rstrip("\"').,;:!?\u201d\u2019]}")
        if not ws:
            return False
        m = _CAPITALISED_TOKEN_RE.fullmatch(ws)
        return m is not None

    cap_flags = [_is_cap_word(w) for w in words]
    phrases: list[tuple[str, tuple[int, int]]] = []
    i = 0
    n = len(words)
    while i < n:
        if not cap_flags[i]:
            i += 1
            continue
        # Extend run of consecutive capitalised words.
        j = i
        while j < n and cap_flags[j]:
            j += 1
        run = list(range(i, j))
        if len(run) >= 2:
            # Emit all sub-n-grams of length 2..ngram_max within the run.
            for L in range(2, min(ngram_max, len(run)) + 1):
                for s in range(len(run) - L + 1):
                    start, end = run[s], run[s] + L
                    raw = " ".join(words[start:end])
                    # Trim punctuation off the edges.
                    clean = raw.strip("\"'()[]{}\u201c\u201d\u2018\u2019,.;:!?")
                    # Drop leading determiners.
                    first, *rest = clean.split()
                    if first in _DISCOVERY_DETERMINER_STOPWORDS and rest:
                        clean = " ".join(rest)
                    parts = clean.split()
                    if len(parts) < 2:
                        continue
                    if parts[0] in _DISCOVERY_GENERIC_LEADS:
                        continue
                    if any(p in _DISCOVERY_PHRASE_BLOCKLIST for p in parts):
                        continue
                    phrases.append((clean, (start, end)))
        i = j
    return phrases, word_spans


# ---------------------------------------------------------------------------
# SAE-feature-allocation heuristic
# ---------------------------------------------------------------------------
# Rough rule of thumb derived from SAE training literature: in a Top-K SAE
# of reasonable width (say 8-16 x d_model), concepts that appear at >= 10
# activation events per million training tokens reliably earn a dedicated
# dictionary slot. Concepts at < 1 event/M almost always get absorbed into
# broader features. The 1-10 range is where SAE width starts to matter.
WORDS_TO_TOKENS_RATIO = 1.3  # English BPE rule of thumb; configurable below.
SAE_LIKELIHOOD_LIKELY_PER_M = 10.0
SAE_LIKELIHOOD_MARGINAL_PER_M = 1.0


def _sae_feature_likelihood(
    per_million_tokens: float,
    *,
    likely_threshold: float = SAE_LIKELIHOOD_LIKELY_PER_M,
    marginal_threshold: float = SAE_LIKELIHOOD_MARGINAL_PER_M,
) -> str:
    if per_million_tokens >= likely_threshold:
        return "likely"
    if per_million_tokens >= marginal_threshold:
        return "marginal"
    return "unlikely"


def discover_concepts(
    text_iter: Iterable[str],
    *,
    top_k: int = 5,
    ngram_max: int = 4,
    min_doc_freq: int = 10,
    min_token_freq: int = 50,
    max_doc_freq_ratio: float = 0.20,
    cooccurrence_window: int = 64,
    max_partners: int = 2,
    max_examples: int = 3,
    context_words: int = 24,
    n_docs_limit: int | None = None,
    words_to_tokens_ratio: float = WORDS_TO_TOKENS_RATIO,
    likely_threshold_per_million: float = SAE_LIKELIHOOD_LIKELY_PER_M,
    marginal_threshold_per_million: float = SAE_LIKELIHOOD_MARGINAL_PER_M,
    n_tokens_actual: int | None = None,
) -> tuple[list[DiscoveredConcept], dict]:
    """Scan a corpus and propose the top-K candidate concepts.

    Heuristic: rank capitalised multi-word phrases (length 2..``ngram_max``)
    by ``log(token_freq) * log(N_docs / doc_freq)``, gated to a sensible
    frequency band (``[min_doc_freq, max_doc_freq_ratio * N_docs]`` documents
    and ``>= min_token_freq`` total occurrences).

    For each top candidate, also surface the most-co-occurring capitalised
    phrase within ``cooccurrence_window`` whitespace tokens. This turns
    singleton candidates into ranked pairs, which is the typical concept
    shape for the unlearning experiment.

    Returns ``(candidates, meta)`` where ``meta`` reports ``n_docs_scanned``
    and ``n_words_scanned``.
    """
    token_freq: Counter[str] = Counter()
    doc_freq: Counter[str] = Counter()
    # Co-occurrence: phrase -> Counter(partner_phrase). Partners include
    # single-word capitalised tokens too, so e.g. "Eiffel Tower" can pair with
    # "Paris". We don't propose single tokens as primary candidates because
    # they tend to be too polysemous, but they're useful as second terms.
    cooc: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict]] = defaultdict(list)

    n_docs = 0
    n_words = 0
    for doc_idx, text in enumerate(text_iter):
        if n_docs_limit is not None and n_docs >= n_docs_limit:
            break
        n_docs += 1
        phrases_with_spans, _word_spans = _extract_capitalised_phrases(text, ngram_max)
        if not phrases_with_spans:
            n_words += len(text.split())
            continue

        # Token / doc freq.
        local_set: set[str] = set()
        for phrase, _span in phrases_with_spans:
            token_freq[phrase] += 1
            local_set.add(phrase)
        for phrase in local_set:
            doc_freq[phrase] += 1

        # Build the partner pool: include single-word capitalised tokens too
        # (e.g. "Paris") so they can be suggested as the second term of a pair.
        words = text.split()
        n_words += len(words)
        single_caps: list[tuple[str, int]] = []  # (token, word_idx)
        for w_idx, w in enumerate(words):
            ws = w.strip("\"'()[]{}\u201c\u201d\u2018\u2019,.;:!?")
            if ws and ws not in _DISCOVERY_DETERMINER_STOPWORDS \
                  and ws not in _DISCOVERY_GENERIC_LEADS \
                  and len(ws) >= 3 and _CAPITALISED_TOKEN_RE.fullmatch(ws):
                single_caps.append((ws, w_idx))

        # Co-occurrence: any pair of distinct items (multi-word phrase or
        # single-word capitalised token) within ``cooccurrence_window`` words.
        # Use phrase start-word index for multi-word phrases and word index for
        # singletons. The pool is the union.
        pool: list[tuple[str, int]] = [(p, s) for p, (s, _e) in phrases_with_spans]
        pool.extend((t, idx) for t, idx in single_caps)
        pool.sort(key=lambda x: x[1])
        for i, (p_i, s_i) in enumerate(pool):
            for j in range(i + 1, len(pool)):
                p_j, s_j = pool[j]
                if s_j - s_i > cooccurrence_window:
                    break
                if p_j == p_i:
                    continue
                # Don't pair a multi-word phrase with one of its own tokens.
                p_i_tokens = set(p_i.split())
                p_j_tokens = set(p_j.split())
                if p_j_tokens.issubset(p_i_tokens) or p_i_tokens.issubset(p_j_tokens):
                    continue
                cooc[p_i][p_j] += 1
                cooc[p_j][p_i] += 1

        # Save a few example contexts per phrase (first occurrence in this doc).
        # Record the doc index so the user can re-locate the example in the
        # source dataset by re-streaming with that offset.
        seen_for_examples: set[str] = set()
        for phrase, (start, end) in phrases_with_spans:
            if phrase in seen_for_examples:
                continue
            if len(examples[phrase]) >= max_examples:
                continue
            lo = max(0, start - context_words)
            hi = min(len(words), end + context_words)
            examples[phrase].append({
                "doc_idx": doc_idx,
                "word_center": (start + end) // 2,
                "context": " ".join(words[lo:hi]),
            })
            seen_for_examples.add(phrase)

    # Filter + score.
    if n_docs == 0:
        return [], {"n_docs_scanned": 0, "n_words_scanned": 0,
                    "n_candidates_before_filter": 0, "n_candidates_after_filter": 0}
    max_doc_freq = int(max_doc_freq_ratio * n_docs)
    candidates: list[DiscoveredConcept] = []
    for phrase, tf in token_freq.items():
        df = doc_freq[phrase]
        if tf < min_token_freq or df < min_doc_freq:
            continue
        if df > max_doc_freq:
            continue
        score = math.log1p(tf) * math.log1p(n_docs / max(df, 1))
        candidates.append(DiscoveredConcept(
            text=phrase,
            n_words=len(phrase.split()),
            token_freq=tf,
            doc_freq=df,
            density=df / n_docs,
            score=score,
        ))
    n_pre_filter = len(token_freq)
    n_post_filter = len(candidates)

    # Deduplicate sub-phrases: if A's word-tuple is a contiguous sub-tuple of
    # B's and they have similar token_freq (B's freq >= 0.8 * A's), drop A.
    # This keeps "Golden Gate Bridge" and drops "Gate Bridge" / "Golden Gate".
    def _is_subphrase(short: list[str], long: list[str]) -> bool:
        if len(short) >= len(long):
            return False
        for i in range(len(long) - len(short) + 1):
            if long[i:i + len(short)] == short:
                return True
        return False

    candidates.sort(key=lambda c: (-c.n_words, -c.score))
    kept_phrases: list[DiscoveredConcept] = []
    for c in candidates:
        c_words = c.text.lower().split()
        absorbed = False
        for kept in kept_phrases:
            kept_words = kept.text.lower().split()
            if _is_subphrase(c_words, kept_words) and kept.token_freq >= 0.8 * c.token_freq:
                # The longer phrase explains most of this candidate's occurrences.
                absorbed = True
                break
            if _is_subphrase(kept_words, c_words) and c.token_freq >= 0.8 * kept.token_freq:
                # The new candidate is a longer, equally-frequent superset; replace.
                kept_phrases = [k for k in kept_phrases if k.text != kept.text]
                kept_phrases.append(c)
                absorbed = True
                break
        if not absorbed:
            kept_phrases.append(c)
    candidates = kept_phrases

    # Final ranking.
    candidates.sort(key=lambda c: -c.score)
    top = candidates[:top_k]

    # Attach partner suggestions, example contexts, and SAE-allocation
    # heuristic (per million tokens, with verdict).
    # Prefer the exact token count from the harvest meta when supplied;
    # otherwise fall back to the words × ratio estimate.
    if n_tokens_actual is not None and n_tokens_actual > 0:
        n_tokens_est = float(n_tokens_actual)
        tokens_source = "harvest_meta"
    else:
        n_tokens_est = max(n_words * words_to_tokens_ratio, 1.0)
        tokens_source = "word_estimate"
    for c in top:
        partners = cooc.get(c.text, Counter()).most_common(max_partners)
        c.suggested_partners = [{"text": p, "cooccurrence_count": cnt} for p, cnt in partners]
        c.example_contexts = list(examples.get(c.text, []))[:max_examples]
        c.per_million_tokens = c.token_freq / n_tokens_est * 1e6
        c.sae_feature_likelihood = _sae_feature_likelihood(
            c.per_million_tokens,
            likely_threshold=likely_threshold_per_million,
            marginal_threshold=marginal_threshold_per_million,
        )

    meta = {
        "n_docs_scanned": n_docs,
        "n_words_scanned": n_words,
        "n_tokens_estimated": int(n_tokens_est),
        "tokens_source": tokens_source,
        "words_to_tokens_ratio": words_to_tokens_ratio,
        "n_candidates_before_filter": n_pre_filter,
        "n_candidates_after_filter": n_post_filter,
        "min_token_freq": min_token_freq,
        "min_doc_freq": min_doc_freq,
        "max_doc_freq_ratio": max_doc_freq_ratio,
        "cooccurrence_window": cooccurrence_window,
        "ngram_max": ngram_max,
        "sae_likely_threshold_per_million": likely_threshold_per_million,
        "sae_marginal_threshold_per_million": marginal_threshold_per_million,
    }
    return top, meta


def discover_concepts_from_shard_dir(
    shard_dir: "Path | str",
    *,
    n_docs_override: int | None = None,
    progress: bool = True,
    **discover_kwargs,
) -> tuple[list[DiscoveredConcept], dict]:
    """Stream exactly the text slice corresponding to a shard dir and run
    :func:`discover_concepts` on it."""
    from datasets import load_dataset

    meta = _load_shard_meta(shard_dir)
    n_docs = n_docs_override if n_docs_override is not None else _shard_dir_n_docs(meta)

    ds = load_dataset(
        meta["dataset_name"], meta.get("dataset_config"),
        split=meta.get("dataset_split", "train"), streaming=True,
    )
    skip = int(meta.get("skip_samples", 0) or 0)
    if skip:
        ds = ds.skip(skip)
    ds = ds.take(n_docs)
    text_field = meta.get("text_field", "text")

    def _it() -> Iterator[str]:
        if progress:
            try:
                from tqdm.auto import tqdm
                bar = tqdm(total=n_docs, desc="[discover]", unit="doc", smoothing=0.05)
            except ImportError:
                bar = None
        else:
            bar = None
        for ex in ds:
            t = ex.get(text_field)
            if t:
                yield t
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()

    # When the shard has a real token count, pass it through so the
    # SAE-likelihood verdict uses the exact harvest tokens rather than the
    # words × ratio estimate.
    n_tokens_actual = meta.get("tokens_written")
    if isinstance(n_tokens_actual, int) and n_tokens_actual > 0:
        discover_kwargs = {**discover_kwargs, "n_tokens_actual": n_tokens_actual}
    candidates, scan_meta = discover_concepts(
        _it(), n_docs_limit=n_docs, **discover_kwargs,
    )
    scan_meta["shard_dir"] = str(shard_dir)
    scan_meta["dataset_name"] = meta["dataset_name"]
    scan_meta["dataset_config"] = meta.get("dataset_config")
    scan_meta["dataset_split"] = meta.get("dataset_split")
    scan_meta["skip_samples"] = skip
    scan_meta["tokens_from_meta"] = n_tokens_actual
    return candidates, scan_meta
