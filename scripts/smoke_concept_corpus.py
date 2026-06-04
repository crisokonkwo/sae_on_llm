"""Offline smoke test for unlearn/concept_data.py.

Uses a mocked Wikipedia fetcher + random-title sampler so no network is
required. Verifies:
  * spec parsing from YAML,
  * sentence splitter and term-matcher behaviour,
  * forget corpus contains only concept-positive sentences,
  * retain corpus contains only concept-negative sentences,
  * statistics + markdown render produce well-formed output,
  * jsonl round-trip preserves data.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from unlearn.concept_data import (
    ConceptDataSpec,
    ConceptProbe,
    build_forget_corpus,
    build_retain_corpus,
    corpus_statistics,
    read_corpus,
    read_probes,
    render_inspection_md,
    sentence_matches,
    split_sentences,
    write_corpus,
    write_probes,
)


# ---------------------------------------------------------------------------
# Synthetic Wikipedia-style text
# ---------------------------------------------------------------------------
_ARTICLES: dict[str, str] = {
    "Eiffel Tower": (
        "== Introduction ==\n"
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
        "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
        "Locally nicknamed La dame de fer, it was constructed from 1887 to 1889. "
        "The Eiffel Tower is the most-visited paid monument in the world. "
        "Today it is widely considered a symbol of Paris and of France.\n\n"
        "== History ==\n"
        "Construction of the Eiffel Tower began in January 1887. It was completed in March 1889."
    ),
    "Champ de Mars": (
        "The Champ de Mars is a large public greenspace located between the Eiffel Tower and the École Militaire. "
        "Originally used for military drills, it has hosted numerous events. "
        "The Eiffel Tower stands at the northwestern end of the Champ de Mars."
    ),
    "Gustave Eiffel": (
        "Alexandre Gustave Eiffel was a French civil engineer. "
        "He is best known for the world-famous Eiffel Tower, built for the 1889 Universal Exposition in Paris. "
        "Eiffel also contributed to the Statue of Liberty's internal framework."
    ),
}
_RANDOM_ARTICLES: dict[str, str] = {
    "Quokka": (
        "The quokka is a small marsupial native to a small corner of Australia. "
        "They are about the size of a domestic cat and are known for their friendly disposition. "
        "Tourists often visit Rottnest Island specifically to see quokkas."
    ),
    "Spaghetti carbonara": (
        "Spaghetti carbonara is a Roman pasta dish made with eggs, hard cheese, cured pork, and black pepper. "
        "It is widely considered an icon of Italian cuisine. "
        "Many regional variations exist but the original uses guanciale, not bacon."
    ),
    "Octopus": (
        "An octopus is a soft-bodied, eight-limbed mollusc of the order Octopoda. "
        "Octopuses are highly intelligent and capable of solving complex puzzles. "
        "They are known for their ability to change colour and texture for camouflage."
    ),
    # One trap: a random article that does mention the concept term, so we
    # can verify the retain builder filters it out.
    "Trap article": (
        "This random article happens to mention the Eiffel Tower in passing. "
        "The rest is unrelated content about coffee brewing."
    ),
}


def _fake_fetch(title: str) -> str:
    if title in _ARTICLES:
        return _ARTICLES[title]
    if title in _RANDOM_ARTICLES:
        return _RANDOM_ARTICLES[title]
    return ""


def _fake_sample(n: int) -> list[str]:
    titles = list(_RANDOM_ARTICLES.keys())
    return titles[:n]


def main() -> None:
    # Confirm helpers behave as advertised.
    sents = split_sentences("Hello world. Second sentence! Third? Yes.")
    assert len(sents) >= 3, sents
    assert sentence_matches("Eiffel Tower is in Paris", ["Eiffel Tower", "Berlin"]) == ["Eiffel Tower"]
    assert sentence_matches("eiffel tower is in paris", ["Eiffel Tower"]) == ["Eiffel Tower"]
    assert sentence_matches("She wrote a tower-related paper", ["Eiffel Tower"]) == []

    # Build a minimal spec inline (YAML round-trip tested below).
    spec = ConceptDataSpec(
        name="eiffel_tower",
        description="smoke test concept",
        positive_required_terms=["Eiffel Tower"],
        positive_optional_terms=["Gustave Eiffel", "Champ de Mars"],
        wikipedia_titles=list(_ARTICLES.keys()),
        target_forget_sentences=20,
        target_retain_sentences=10,
        retain_source="wikipedia_random",
        min_sentence_chars=20, max_sentence_chars=400,
        probes=[
            ConceptProbe(prompt="The Eiffel Tower is in",
                         paraphrases=["You can find the Eiffel Tower in"],
                         expected_completion=" Paris"),
            ConceptProbe(prompt="Who designed the Eiffel Tower?",
                         paraphrases=["The Eiffel Tower was designed by"],
                         expected_completion=" Gustave Eiffel"),
        ],
    )

    forget = build_forget_corpus(spec, fetch_text=_fake_fetch, progress=False)
    assert len(forget) > 0
    for s in forget:
        assert s.is_positive
        assert s.matched_terms, f"forget sentence has no matched terms: {s.text!r}"
        assert any(t in s.text or t.lower() in s.text.lower() for t in s.matched_terms)
    assert any("Eiffel Tower" in s.text for s in forget)
    print(f"[smoke] forget: {len(forget)} sentences")

    retain = build_retain_corpus(
        spec, fetch_text=_fake_fetch, sample_titles=_fake_sample, progress=False,
    )
    for s in retain:
        assert not s.is_positive
        assert not sentence_matches(s.text, spec.all_positive_terms), \
            f"retain sentence leaked concept term: {s.text!r}"
    # The trap article's first sentence should have been dropped.
    assert all("Eiffel Tower" not in s.text for s in retain)
    print(f"[smoke] retain: {len(retain)} sentences")

    # Statistics + render
    stats = corpus_statistics(forget, term_counts_for=spec.all_positive_terms)
    assert stats.n_sentences == len(forget)
    assert stats.n_chars_mean > 0
    assert stats.term_match_counts.get("Eiffel Tower", 0) >= 1
    retain_stats = corpus_statistics(retain)
    md = render_inspection_md(spec, forget, retain,
                              forget_stats=stats, retain_stats=retain_stats)
    assert "# Concept dataset: eiffel_tower" in md
    assert "## Forget corpus" in md and "## Retain corpus" in md
    print(f"[smoke] markdown: {len(md.splitlines())} lines")

    # Round-trip jsonl I/O
    tmp = Path(tempfile.mkdtemp(prefix="concept_data_smoke_"))
    write_corpus(forget, tmp / "forget.jsonl")
    write_corpus(retain, tmp / "retain.jsonl")
    write_probes(spec.probes, tmp / "probes.jsonl")
    forget_rt = read_corpus(tmp / "forget.jsonl")
    retain_rt = read_corpus(tmp / "retain.jsonl")
    probes_rt = read_probes(tmp / "probes.jsonl")
    assert len(forget_rt) == len(forget) and forget_rt[0].text == forget[0].text
    assert len(retain_rt) == len(retain)
    assert len(probes_rt) == len(spec.probes)
    assert probes_rt[0].paraphrases == spec.probes[0].paraphrases

    # YAML round-trip via ConceptDataSpec.from_yaml
    yaml_path = tmp / "spec.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "name": "smoke", "positive_required_terms": ["X"],
        "wikipedia_titles": ["A"], "probes": [{"prompt": "P", "paraphrases": ["P2"]}],
    }, sort_keys=False))
    spec2 = ConceptDataSpec.from_yaml(yaml_path)
    assert spec2.name == "smoke"
    assert spec2.positive_required_terms == ["X"]
    assert len(spec2.probes) == 1
    assert spec2.probes[0].paraphrases == ["P2"]

    print("[smoke] ALL OK")


if __name__ == "__main__":
    main()
