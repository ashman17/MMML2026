from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fvpg import Claim, Derivation, Frame, ProofChecker, ProofGraph, Status


graph = ProofGraph()
graph.add_claim(Claim("chair_behind_bin", "behind", ("chair", "dustbin"), Frame.GLOBAL, status=Status.SUPPORTED))
graph.add_claim(Claim("bin_left_curtain", "left_of", ("dustbin", "curtain"), Frame.GLOBAL, status=Status.SUPPORTED))
graph.add_claim(Claim("chair_faces_curtain", "facing", ("chair", "curtain"), Frame.GLOBAL))
graph.add_claim(Claim("curtain_front_left", "front_left_of", ("curtain", "seated_observer"), Frame.OBJECT))

# This is deliberately invalid: relative positions cannot establish orientation.
graph.add_derivation(Derivation(
    "plausible_but_invalid_bridge",
    "position_implies_orientation",
    ("chair_behind_bin", "bin_left_curtain"),
    "chair_faces_curtain",
))

result = ProofChecker().check(graph)
print("certificate_valid:", result.valid)
print("errors:")
for error in result.errors:
    print(" -", error)
print("unresolved_claims:", sorted(result.unresolved_claims))
