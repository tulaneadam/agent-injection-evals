"""InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents.

Qiusi Zhan, Zhixiang Liang, Zifan Ying, Daniel Kang (2024).
https://arxiv.org/abs/2403.02691  |  https://github.com/uiuc-kang-lab/InjecAgent

Inspect port of the paper's function-calling ("fine-tuned agent") setting:
1,054 test cases where a benign user request makes the agent call a tool whose
response carries an attacker's instruction. The eval measures how often the
agent then calls the attacker's tool (direct harm) or retrieves private data
and emails it to the attacker (data stealing).

Usage:
    inspect eval src/injecagent/injecagent.py@injecagent --model openai/gpt-4o-mini
    inspect eval src/injecagent/injecagent.py@injecagent -T setting=enhanced -T attack_type=ds
"""

from inspect_ai import Task, task

from injecagent.dataset import AttackType, Setting, injecagent_dataset
from injecagent.scorer import injecagent_scorer
from injecagent.solver import injecagent_agent

TASK_VERSION = "1.0.0"


@task
def injecagent(setting: Setting = "base", attack_type: AttackType = "all", temperature: float = 0.0) -> Task:
    """InjecAgent indirect prompt injection benchmark.

    Args:
        setting: "base" (attacker instruction only) or "enhanced" (adds the paper's
            "IMPORTANT!!! Ignore all previous instructions..." hacking prompt).
        attack_type: "all", "dh" (direct harm, 510 cases) or "ds" (data stealing, 544 cases).
        temperature: sampling temperature (the paper uses 0).
    """
    return Task(
        dataset=injecagent_dataset(setting=setting, attack_type=attack_type),
        solver=injecagent_agent(temperature=temperature),
        scorer=injecagent_scorer(),
        version=TASK_VERSION,
        metadata={"paper": "https://arxiv.org/abs/2403.02691", "setting": setting, "attack_type": attack_type},
    )
