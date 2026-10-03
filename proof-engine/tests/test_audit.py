import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_rationales import conclusion_clause, relation_mentions


class RationaleAuditTest(unittest.TestCase):
    def test_diagonal_does_not_create_nested_cardinal_mentions(self) -> None:
        self.assertEqual(relation_mentions("Therefore it is at the front left."), {"front_left_of"})

    def test_chair_curtain_conclusion_is_front(self) -> None:
        thought = (
            "The chair is rotated. By analyzing both images, the curtain happens "
            "to be positioned at a 45-degree angle, and therefore is directly in front."
        )
        clause = conclusion_clause(thought)
        self.assertTrue(clause.lower().startswith("therefore"))
        self.assertEqual(relation_mentions(clause), {"front_of"})

    def test_composed_phrase_becomes_diagonal(self) -> None:
        self.assertEqual(
            relation_mentions("The object is ahead and to the right."),
            {"front_right_of"},
        )

    def test_negated_direction_is_undetermined(self) -> None:
        self.assertEqual(
            relation_mentions("It is impossible to determine what is on the east side."),
            {"undetermined"},
        )
