"""
tasks3d.py — ordered subgoals (navigate/pick/place) for the 3D harness.

This is a close adaptation of the 2D tasks.py — same Subgoal/TaskRunner
shape, same rule-based + LLM-grounded parsing — because the actual hard
problem (grounding "the can" against real asset names, sequencing
navigate->pick->navigate->place) isn't dimension-specific. What's
different: target positions are 3D now, but nav_agent_3d only cares about
the (x, z) floor projection, so _target_pos just returns the full 3D
position and callers slice what they need.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

import nav_agent_3d as nav


@dataclass
class Subgoal:
    kind: str
    target: str
    container: Optional[str] = None
    done: bool = False


def _target_pos(scene, name):
    if name in scene.markers:
        return scene.markers[name]
    if name in scene.assets:
        return tuple(scene.assets[name].pos)
    raise KeyError(f"task references unknown target {name!r}; "
                    f"known assets: {list(scene.assets)}, markers: {list(scene.markers)}")


def _ground_name(word, scene):
    word = word.lower().strip()
    candidates = list(scene.assets) + list(scene.markers)
    for c in candidates:
        if c.lower() == word:
            return c
    for c in candidates:
        if word in c.lower() or c.lower() in word:
            return c
    word_tokens = set(re.split(r"[_\s]+", word))
    best, best_overlap = None, 0
    for c in candidates:
        overlap = len(word_tokens & set(re.split(r"[_\s]+", c.lower())))
        if overlap > best_overlap:
            best, best_overlap = c, overlap
    return best


_PICK_PLACE_RE = re.compile(
    r"pick(?:\s+up)?\s+(?:the\s+)?(.+?)\s+and\s+(?:place|put|drop)\s+"
    r"(?:it\s+)?in(?:to)?\s+(?:the\s+)?(.+)", re.I)


def _rule_based_parse(task_text, scene):
    m = _PICK_PLACE_RE.search(task_text)
    if m:
        item = _ground_name(m.group(1).strip(), scene)
        container = _ground_name(m.group(2).strip(), scene)
        if item and container:
            return [
                Subgoal("navigate", item),
                Subgoal("pick", item),
                Subgoal("navigate", container),
                Subgoal("place", item, container),
            ]
    if scene.goal is not None:
        return [Subgoal("navigate", scene.goal.name)]
    raise ValueError(f"couldn't parse task: {task_text!r}")


def _llm_parse(task_text, scene, model="gpt-4o"):
    from synthesizer3d import _call_llm
    names = list(scene.to_state().keys()) + list(scene.markers.keys())
    system = (
        "You convert a short task instruction into a JSON list of subgoals "
        "for a 3D agent. Each subgoal is an object with 'kind' one of "
        "'navigate', 'pick', 'place'. 'place' also has a 'container' field. "
        "'target' and 'container' MUST be exactly one of these existing "
        f"names, chosen by matching meaning: {names}. "
        'Pick-and-place -> [{"kind":"navigate","target":"<item>"},'
        '{"kind":"pick","target":"<item>"},{"kind":"navigate","target":"<container>"},'
        '{"kind":"place","target":"<item>","container":"<container>"}]. '
        "Reply with ONLY the JSON list, no prose, no markdown fences."
    )
    raw = _call_llm(system, f'Task: "{task_text}"', model=model)
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    data = json.loads(raw)
    return [Subgoal(d["kind"], d["target"], d.get("container")) for d in data]


def parse_task(task_text, scene, use_mock=False, model="gpt-4o"):
    if use_mock:
        return _rule_based_parse(task_text, scene)
    try:
        return _llm_parse(task_text, scene, model=model)
    except Exception:
        return _rule_based_parse(task_text, scene)


class TaskRunner:
    def __init__(self, subgoals):
        self.subgoals = subgoals
        self.idx = 0

    @property
    def current(self):
        return self.subgoals[self.idx] if self.idx < len(self.subgoals) else None

    @property
    def is_complete(self):
        return self.idx >= len(self.subgoals)

    def current_target_pos(self, scene):
        sg = self.current
        return None if sg is None else _target_pos(scene, sg.target)

    def try_advance(self, scene, arrival_threshold=34):
        sg = self.current
        if sg is None:
            return False
        pos = _target_pos(scene, sg.target)
        if not nav.reached(scene, pos, threshold=arrival_threshold):
            return False
        if sg.kind == "navigate":
            pass
        elif sg.kind == "pick":
            scene.pick(scene.assets[sg.target])
        elif sg.kind == "place":
            scene.place(scene.assets[sg.container])
        else:
            raise ValueError(sg.kind)
        sg.done = True
        self.idx += 1
        nav.reset_stuck_history()
        return True

    def describe(self):
        lines = []
        for i, sg in enumerate(self.subgoals):
            marker = "x" if sg.done else (">" if i == self.idx else " ")
            extra = f" -> {sg.container}" if sg.container else ""
            lines.append(f"[{marker}] {sg.kind} {sg.target}{extra}")
        return lines