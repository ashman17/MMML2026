from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from .models import Claim, Frame, ProofGraph


OPTION_RE = re.compile(
    r"(?:^|,\s*)([A-Z]):\s*(.*?)(?=,\s*[A-Z]:|$)", re.DOTALL
)


@dataclass(frozen=True)
class AnswerHypothesis:
    option: str
    text: str
    conclusion_claim_id: str
    required_claim_ids: tuple[str, ...]


@dataclass
class CompiledQuestion:
    question_id: int
    category: str
    stem: str
    target: str
    reference: str
    frame: Frame
    graph: ProofGraph
    hypotheses: list[AnswerHypothesis]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "stem": self.stem,
            "target": self.target,
            "reference": self.reference,
            "frame": self.frame.value,
            "claims": {
                claim_id: {
                    **asdict(claim),
                    "frame": claim.frame.value,
                    "status": claim.status.value,
                }
                for claim_id, claim in self.graph.claims.items()
            },
            "hypotheses": [asdict(hypothesis) for hypothesis in self.hypotheses],
            "warnings": self.warnings,
        }


class MMSIProofCompiler:
    """Compiles MMSI positional questions into unverified proof skeletons.

    The compiler is intentionally conservative: linguistic extraction creates
    claims to test, never evidence that those claims are true.
    """

    RELATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"^(front|upper)\s*[- ]?left(?: corner)?$"), "front_left_of"),
        (re.compile(r"^(front|upper)\s*[- ]?right(?: corner)?$"), "front_right_of"),
        (re.compile(r"^(back|rear|lower|left rear)\s*[- ]?left(?: corner)?$"), "back_left_of"),
        (re.compile(r"^(back|rear|lower|right rear)\s*[- ]?right(?: corner)?$"), "back_right_of"),
        (re.compile(r"^(directly )?(to |on )?(the |your )?left$|^west$"), "left_of"),
        (re.compile(r"^(directly )?(to |on )?(the |your )?right$|^east$"), "right_of"),
        (re.compile(r"^(directly )?(in )?front(?: of you)?$|^forward$|^north$"), "front_of"),
        (re.compile(r"^(directly )?(behind|back|rear)(?: you)?$|^backward$|^south$"), "behind"),
        (re.compile(r"^northwest$"), "front_left_of"),
        (re.compile(r"^northeast$"), "front_right_of"),
        (re.compile(r"^southwest$"), "back_left_of"),
        (re.compile(r"^southeast$"), "back_right_of"),
        (re.compile(r"^up$|^above$"), "above"),
        (re.compile(r"^down$|^below$"), "below"),
        (re.compile(r"^cannot be determined$|^unable to determine$"), "undetermined"),
    )

    POSITIONAL_MARKERS = ("Positional Relationship",)
    GEOGRAPHIC_RELATIONS = {
        "north": "north_of", "south": "south_of",
        "east": "east_of", "west": "west_of",
        "northeast": "northeast_of", "northwest": "northwest_of",
        "southeast": "southeast_of", "southwest": "southwest_of",
    }

    def compile_record(self, record: dict[str, Any]) -> CompiledQuestion:
        raw_question = str(record["question"])
        stem, options = self._split_question(raw_question)
        target, reference, extraction_warnings = self._extract_entities(stem, str(record["category"]))
        frame = self._infer_frame(str(record["category"]), stem, options)

        graph = ProofGraph()
        graph.add_claim(Claim("target_grounded", "grounded", (target,), Frame.IMAGE))
        graph.add_claim(Claim("reference_frame", "frame_resolved", (reference,), frame))

        prerequisites = ["target_grounded", "reference_frame"]
        if "sitting" in stem.lower() or "sit on" in stem.lower():
            graph.add_claim(Claim(
                "observer_orientation",
                "facing_direction_resolved",
                (reference,),
                frame,
            ))
            prerequisites.append("observer_orientation")

        hypotheses: list[AnswerHypothesis] = []
        warnings = list(extraction_warnings)
        for label, text in options:
            predicate = self.normalize_relation(text, geographic=frame is Frame.GEOGRAPHIC)
            if predicate is None:
                warnings.append(f"option {label} is not a supported spatial relation: {text!r}")
                continue
            claim_id = f"option_{label.lower()}"
            graph.add_claim(Claim(claim_id, predicate, (target, reference), frame))
            hypotheses.append(AnswerHypothesis(
                label,
                text,
                claim_id,
                tuple((*prerequisites, claim_id)),
            ))

        if len(hypotheses) != len(options):
            warnings.append("question is only partially compiled")

        return CompiledQuestion(
            int(record["id"]),
            str(record["category"]),
            stem,
            target,
            reference,
            frame,
            graph,
            hypotheses,
            warnings,
        )

    @classmethod
    def normalize_relation(cls, text: str, geographic: bool = False) -> str | None:
        normalized = re.sub(r"\s+", " ", text.strip().rstrip(".").lower())
        if geographic and normalized in cls.GEOGRAPHIC_RELATIONS:
            return cls.GEOGRAPHIC_RELATIONS[normalized]
        for pattern, predicate in cls.RELATION_PATTERNS:
            if pattern.fullmatch(normalized):
                return predicate
        return None

    @staticmethod
    def _split_question(question: str) -> tuple[str, list[tuple[str, str]]]:
        if "Options:" not in question:
            return question.strip(), []
        stem, option_text = question.split("Options:", 1)
        return stem.strip(), [
            (label, text.strip().rstrip("."))
            for label, text in OPTION_RE.findall(option_text.strip())
        ]

    @staticmethod
    def _extract_entities(stem: str, category: str) -> tuple[str, str, list[str]]:
        patterns = (
            re.compile(
                r"When you (?:are sitting|sit) on (?:the )?(?P<reference>.+?),\s*"
                r"where is (?:the )?(?P<target>.+?) (?:located )?(?:in relation|relative) to you\?*$",
                re.IGNORECASE,
            ),
            re.compile(
                r"When you take the photo in (?P<reference>Image \d+),\s*"
                r"where is (?:the )?(?P<target>.+?) located relative to you\?*$",
                re.IGNORECASE,
            ),
            re.compile(
                r"When you enter (?:from|through) (?P<reference>.+?),\s*"
                r"in which direction is (?:the )?(?P<target>.+?) located relative to you\?*$",
                re.IGNORECASE,
            ),
        )
        for pattern in patterns:
            match = pattern.match(stem)
            if match:
                return (
                    match.group("target").strip(),
                    match.group("reference").strip(),
                    [],
                )

        # High-frequency MMSI forms. These extract candidate entities but do
        # not assert that the parse is visually or semantically correct.
        generic_patterns = (
            re.compile(
                r"where (?:is|was) (?:the )?(?P<target>.+?) (?:located )?"
                r"(?:in relation|relative) to (?P<reference>.+?)\?*$",
                re.IGNORECASE,
            ),
            re.compile(
                r"in which direction (?:is|was) (?:the )?(?P<target>.+?) (?:located )?"
                r"(?:in relation to|relative to|from) (?P<reference>.+?)\?*$",
                re.IGNORECASE,
            ),
            re.compile(
                r"what is (?:the )?position of (?:the )?(?P<target>.+?) "
                r"relative to (?:the )?(?P<reference>.+?)\?*$",
                re.IGNORECASE,
            ),
            re.compile(
                r"in which direction relative to (?P<reference>.+?) is "
                r"(?:the )?(?P<target>.+?) (?:likely )?located\?*$",
                re.IGNORECASE,
            ),
        )
        for pattern in generic_patterns:
            match = pattern.search(stem)
            if match:
                target = MMSIProofCompiler._clean_entity(match.group("target"))
                reference = MMSIProofCompiler._clean_entity(match.group("reference"))
                if reference.lower() in {"you", "me", "it"}:
                    reference = "observer_at_question_pose"
                return target, reference, ["generic linguistic parse; requires semantic verification"]

        # Camera-camera questions have stable semantic roles even when phrased
        # without an explicit target noun phrase.
        if "Cam.–Cam." in category:
            image_numbers = re.findall(r"(?:photo|picture|image)\s*(\d+|first|second)", stem, re.IGNORECASE)
            if len(image_numbers) >= 2:
                return (
                    f"camera_pose_{image_numbers[-1].lower()}",
                    f"camera_pose_{image_numbers[0].lower()}",
                    ["camera-role parse; requires temporal verification"],
                )

        # Partial parsing is still useful: grounding can be requested while the
        # reference frame remains explicitly unresolved.
        target_patterns = (
            re.compile(r"where (?:is|was) (?:the )?(?P<target>.+?)(?: located)?\?*$", re.IGNORECASE),
            re.compile(r"in which direction (?:is|was) (?:the )?(?P<target>.+?)(?: located)?\?*$", re.IGNORECASE),
        )
        for pattern in target_patterns:
            match = pattern.search(stem)
            if match:
                return (
                    MMSIProofCompiler._clean_entity(match.group("target")),
                    "reference_entity",
                    ["reference extraction unresolved"],
                )
        return "target_entity", "reference_entity", ["target and reference extraction unresolved"]

    @staticmethod
    def _clean_entity(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text.strip().rstrip(".?"))
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    @staticmethod
    def _infer_frame(
        category: str, stem: str, options: list[tuple[str, str]] | None = None
    ) -> Frame:
        normalized_options = {
            re.sub(r"\s+", " ", text.strip().rstrip(".").lower())
            for _, text in (options or [])
        }
        if normalized_options and normalized_options <= set(MMSIProofCompiler.GEOGRAPHIC_RELATIONS):
            return Frame.GEOGRAPHIC
        if "Cam." in category or "take the photo" in stem:
            return Frame.CAMERA
        if "sitting" in stem.lower() or "sit on" in stem.lower():
            return Frame.OBJECT
        return Frame.GLOBAL
