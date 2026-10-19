"""Reference Track B baseline: the starter-kit Qwen2.5-VL-3B-Instruct system.

Copy this file and models.txt to submissions/<team_id>/track_b/ in a pull request.
"""

from evaluator_cli import QwenISUSystem


def create_system(device):
    return QwenISUSystem(device=device)
