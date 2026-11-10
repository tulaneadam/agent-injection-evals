"""Build Inspect samples from the InjecAgent test cases."""

from __future__ import annotations

from typing import List, Literal

from inspect_ai.dataset import MemoryDataset, Sample

from .data import load_json

Setting = Literal["base", "enhanced"]
AttackType = Literal["all", "dh", "ds"]

ATTACK_NAMES = {"dh": "direct_harm", "ds": "data_stealing"}


def injecagent_dataset(setting: Setting = "base", attack_type: AttackType = "all") -> MemoryDataset:
    if setting not in ("base", "enhanced"):
        raise ValueError("setting must be 'base' or 'enhanced'")
    if attack_type not in ("all", "dh", "ds"):
        raise ValueError("attack_type must be 'all', 'dh' or 'ds'")
    attacks: List[str] = ["dh", "ds"] if attack_type == "all" else [attack_type]
    samples: List[Sample] = []
    for attack in attacks:
        for index, case in enumerate(load_json(f"test_cases_{attack}_{setting}.json")):
            samples.append(
                Sample(
                    id=f"{attack}-{setting}-{index:04d}",
                    input=case["User Instruction"],
                    # The target is the attacker tool whose invocation counts as a successful attack.
                    target=case["Attacker Tools"][0],
                    metadata={
                        "attack": attack,
                        "attack_name": ATTACK_NAMES[attack],
                        "setting": setting,
                        "attack_type_label": case["Attack Type"],
                        "user_tool": case["User Tool"],
                        "user_instruction": case["User Instruction"],
                        "tool_parameters": case["Tool Parameters"],
                        "tool_response": case["Tool Response"],
                        "attacker_tools": case["Attacker Tools"],
                        "attacker_instruction": case["Attacker Instruction"],
                        "expected_achievements": case["Expected Achievements"],
                    },
                )
            )
    return MemoryDataset(samples, name=f"injecagent_{setting}_{attack_type}")
