from __future__ import annotations

from dataclasses import dataclass, field

from .models import Claim, Derivation, ProofGraph, Status


INVERSE_PREDICATES = {
    "left_of": "right_of",
    "right_of": "left_of",
    "front_of": "behind",
    "behind": "front_of",
    "above": "below",
    "below": "above",
}


@dataclass
class CheckResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    supported_claims: set[str] = field(default_factory=set)
    unresolved_claims: set[str] = field(default_factory=set)


class ProofChecker:
    """Checks proof structure and propagates only whitelisted spatial rules."""

    def check(self, graph: ProofGraph) -> CheckResult:
        errors = self._validate_references(graph)
        if errors:
            return CheckResult(False, errors)

        supported = {
            claim.id for claim in graph.claims.values()
            if self._direct_status(graph, claim) is Status.SUPPORTED
        }

        changed = True
        while changed:
            changed = False
            for derivation in graph.derivations:
                error = self._validate_rule(graph, derivation)
                if error:
                    errors.append(f"{derivation.id}: {error}")
                    continue
                if all(premise in supported for premise in derivation.premises):
                    if derivation.conclusion not in supported:
                        supported.add(derivation.conclusion)
                        changed = True

        unresolved = set(graph.claims) - supported
        return CheckResult(not errors, errors, supported, unresolved)

    @staticmethod
    def _direct_status(graph: ProofGraph, claim: Claim) -> Status:
        statuses = [graph.evidence[eid].status for eid in claim.evidence_ids]
        if Status.CONTRADICTED in statuses:
            return Status.CONTRADICTED
        if Status.SUPPORTED in statuses:
            return Status.SUPPORTED
        return claim.status

    @staticmethod
    def _validate_references(graph: ProofGraph) -> list[str]:
        errors: list[str] = []
        for claim in graph.claims.values():
            for evidence_id in claim.evidence_ids:
                if evidence_id not in graph.evidence:
                    errors.append(f"{claim.id}: unknown evidence {evidence_id}")
        for derivation in graph.derivations:
            missing = [c for c in (*derivation.premises, derivation.conclusion) if c not in graph.claims]
            if missing:
                errors.append(f"{derivation.id}: unknown claims {missing}")
        return errors

    def _validate_rule(self, graph: ProofGraph, derivation: Derivation) -> str | None:
        if derivation.rule == "identity":
            if len(derivation.premises) != 1:
                return "identity requires one premise"
            premise = graph.claims[derivation.premises[0]]
            conclusion = graph.claims[derivation.conclusion]
            if not self._same_claim(premise, conclusion):
                return "identity premise and conclusion differ"
            return None

        if derivation.rule == "inverse_relation":
            if len(derivation.premises) != 1:
                return "inverse_relation requires one premise"
            premise = graph.claims[derivation.premises[0]]
            conclusion = graph.claims[derivation.conclusion]
            expected = INVERSE_PREDICATES.get(premise.predicate)
            if expected is None:
                return f"no inverse registered for {premise.predicate}"
            if conclusion.predicate != expected:
                return f"expected predicate {expected}"
            if conclusion.arguments != tuple(reversed(premise.arguments)):
                return "inverse relation must reverse arguments"
            if conclusion.frame != premise.frame:
                return "inverse relation cannot change coordinate frame"
            return None

        return f"unregistered rule {derivation.rule!r}"

    @staticmethod
    def _same_claim(left: Claim, right: Claim) -> bool:
        return (
            left.predicate == right.predicate
            and left.arguments == right.arguments
            and left.frame == right.frame
            and left.views == right.views
        )
