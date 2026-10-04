"""Claims a reference trace adds beyond the question.

A cue counts only when the stem and options do not already state it. Step F's
trace cues stay as they are: those mark presence inside the trace, including
words the question already used. Nothing here is a shortcut feature.
"""

from __future__ import annotations

import re

from entities import IMAGE_NOUNS, SKIP_HEADS
from lexicon import image_refs
from traces import content_lemmas

# "viewpoint" and "camera" are landmarks of a relation ("behind the first
# viewpoint"). Direction words, image words, and query words stay excluded.
_LANDMARK_EXCEPTIONS = {"viewpoint", "camera"}
_SIMPLE_RELATIONS = {
    "behind": "behind",
    "beside": "beside",
    "between": "between",
    "above": "above",
    "below": "below",
    "under": "under",
    "underneath": "under",
    "beneath": "under",
    "over": "over",
    "opposite": "opposite",
    "near": "near",
}
_TO_RELATIONS = {"next": "next to", "adjacent": "adjacent to", "close": "close to"}
_COMPARATIVES = (
    "closer", "farther", "further", "larger", "smaller", "higher", "lower", "taller",
    "shorter", "longer", "wider", "narrower", "bigger", "greater",
)
_VIEWER = re.compile(
    r"\b(?:fac(?:e|es|ed|ing)|perspective|viewpoint|camera|"
    r"compar(?:ing|ed) with|compare with)\b",
    re.I,
)
_IDENTITY = (
    ("not the same", re.compile(r"\bnot the same\b", re.I)),
    ("same as", re.compile(r"\bsame as\b", re.I)),
    ("overlap", re.compile(r"\boverlap\w*\b", re.I)),
)
_ROUTE_TEXT = (
    ("go straight", re.compile(r"\bgo straight\b", re.I)),
    ("cannot", re.compile(r"\bcannot\b|\bcan't\b", re.I)),
)


def _as_index(value, n_images: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    if value in {"last", "final"}:
        return int(n_images) if n_images else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _landmark_lemma(token) -> str | None:
    if token.pos_ == "PRON":
        return None
    lemma = token.lemma_.lower()
    if not lemma.isalpha() or lemma in IMAGE_NOUNS:
        return None
    if lemma in SKIP_HEADS and lemma not in _LANDMARK_EXCEPTIONS:
        return None
    if token.pos_ not in {"NOUN", "PROPN"} and lemma not in _LANDMARK_EXCEPTIONS:
        return None
    return lemma


def _object_tokens(prep):
    tokens = []
    for child in prep.children:
        if child.dep_ != "pobj":
            continue
        tokens.append(child)
        tokens.extend(child.conjuncts)
    return tokens


def _add_landmarks(pairs: set[tuple[str, str]], relation: str, prep) -> None:
    for token in _object_tokens(prep):
        lemma = _landmark_lemma(token)
        if lemma:
            pairs.add((relation, lemma))


def _to_preps(token):
    found = [child for child in token.children if child.lemma_.lower() == "to" and child.dep_ == "prep"]
    if token.dep_ in {"advmod", "acomp", "amod", "advcl"}:
        for child in token.head.children:
            if child.lemma_.lower() == "to" and child.dep_ == "prep" and child.i > token.i and child not in found:
                found.append(child)
    return found


def spatial_pairs(doc) -> set[tuple[str, str]]:
    """(relation, landmark lemma) pairs. A bare direction word has no landmark."""
    pairs: set[tuple[str, str]] = set()
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma == "in" and token.dep_ == "prep":
            for front in token.children:
                if front.lemma_.lower() != "front" or front.dep_ != "pobj":
                    continue
                for of in front.children:
                    if of.lemma_.lower() == "of" and of.dep_ == "prep":
                        _add_landmarks(pairs, "in front of", of)
        if lemma == "across":
            for child in token.children:
                if child.lemma_.lower() == "from" and child.dep_ == "prep":
                    _add_landmarks(pairs, "across from", child)
        if lemma in _TO_RELATIONS:
            for prep in _to_preps(token):
                _add_landmarks(pairs, _TO_RELATIONS[lemma], prep)
        if lemma in _SIMPLE_RELATIONS and token.dep_ in {"prep", "conj"}:
            _add_landmarks(pairs, _SIMPLE_RELATIONS[lemma], token)
    return pairs


def added_spatial_pairs(trace_doc, question_doc) -> set[tuple[str, str]]:
    return spatial_pairs(trace_doc) - spatial_pairs(question_doc)


def comparative_words(text: str) -> set[str]:
    return {word for word in _COMPARATIVES if re.search(rf"\b{word}\b", text, flags=re.I)}


def added_comparatives(trace_text: str, question_text: str) -> set[str]:
    """Comparative words the trace uses and the question does not.

    "Between" is a spatial relation, not a comparison. Repeating "closer" when
    the question asks "which is closer" is not an addition; a different
    comparative word still is.
    """
    return comparative_words(trace_text) - comparative_words(question_text)


def _viewer_images(doc, n_images: int | None) -> set[int]:
    found = set()
    for sent in doc.sents:
        if not _VIEWER.search(sent.text):
            continue
        for _, index in image_refs(sent.text):
            normalized = _as_index(index, n_images)
            if normalized is not None:
                found.add(normalized)
    return found


def facing_complements(doc) -> set[str]:
    """Nouns a viewer faces, or the complement of 'perspective/viewpoint of'."""
    found = set()
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma == "face" and token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ in {"dobj", "obj", "attr", "pobj"}:
                    landmark = _landmark_lemma(child)
                    if landmark:
                        found.add(landmark)
                elif child.dep_ == "prep" and child.lemma_.lower() in {"toward", "towards"}:
                    for obj in _object_tokens(child):
                        landmark = _landmark_lemma(obj)
                        if landmark:
                            found.add(landmark)
        elif lemma in {"perspective", "viewpoint"}:
            for child in token.children:
                if child.dep_ == "prep" and child.lemma_.lower() == "of":
                    for obj in _object_tokens(child):
                        landmark = _landmark_lemma(obj)
                        if landmark:
                            found.add(landmark)
    return found


def added_viewpoint(trace_doc, question_doc, anchor=None, n_images: int | None = None) -> dict:
    """A viewer shift the question does not already specify.

    Question viewer images are the Step A anchor plus image indexes in a
    facing, perspective, or viewpoint clause. Other "Figure N" mentions are
    not viewer shifts. A new facing or perspective complement counts as well.
    """
    question_images = _viewer_images(question_doc, n_images)
    anchor_index = _as_index(anchor, n_images)
    if anchor_index is not None:
        question_images.add(anchor_index)
    images = _viewer_images(trace_doc, n_images) - question_images
    complements = facing_complements(trace_doc) - facing_complements(question_doc)
    return {"images": images, "complements": complements, "added": bool(images or complements)}


def _route_cues(doc) -> set[str]:
    found = {name for name, pattern in _ROUTE_TEXT if pattern.search(doc.text)}
    for token in doc:
        if (token.pos_ == "VERB" and token.lemma_.lower() in {"turn", "walk"}
                and token.dep_ not in {"amod", "compound", "npadvmod"}):
            found.add(token.lemma_.lower())
    return found


def _identity_cues(doc) -> set[str]:
    found = set()
    for sent in doc.sents:
        if not content_lemmas(sent):
            continue
        text = sent.text
        not_the_same = False
        for name, pattern in _IDENTITY:
            if name == "same as" and not_the_same:
                continue
            if pattern.search(text):
                found.add(name)
                not_the_same = not_the_same or name == "not the same"
    return found


def added_reasoning(trace_doc, question_doc) -> set[str]:
    """Identity or route claims the question does not state.

    Numbered lists and conclusion words do not count. A route word has to be
    the verb "turn" or "walk", or the phrases "go straight" and "cannot", and
    the identity sentence has to contain a content noun. "Walking area" is a
    noun phrase, not a route.
    """
    identity = _identity_cues(trace_doc) - _identity_cues(question_doc)
    # A route cue counts when the question lacks that cue and some trace
    # sentence with a content noun carries it.
    route = set()
    question_routes = _route_cues(question_doc)
    for sent in trace_doc.sents:
        if not content_lemmas(sent):
            continue
        for name in _route_cues(sent) - question_routes:
            route.add(name)
    return identity | route


def addition_record(trace_doc, question_doc, anchor=None, n_images: int | None = None) -> dict:
    spatial = added_spatial_pairs(trace_doc, question_doc)
    comparisons = added_comparatives(trace_doc.text, question_doc.text)
    viewpoint = added_viewpoint(trace_doc, question_doc, anchor=anchor, n_images=n_images)
    reasoning = added_reasoning(trace_doc, question_doc)
    return {
        "add_spatial": bool(spatial),
        "spatial_pairs": sorted(spatial),
        "add_viewpoint": viewpoint["added"],
        "viewpoint_images": sorted(viewpoint["images"]),
        "viewpoint_complements": sorted(viewpoint["complements"]),
        "add_comparison": bool(comparisons),
        "comparison_words": sorted(comparisons),
        "add_reasoning": bool(reasoning),
        "reasoning_cues": sorted(reasoning),
    }
