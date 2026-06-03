import re
from dataclasses import dataclass
from typing import Iterable, Dict, Tuple, List, Optional

from torch import nn, Tensor


@dataclass
class RegexRule:
    pattern: re.Pattern
    repl: str

def rr(p: str, r: str) -> RegexRule:
    """
    Create a RegexRule from pattern and replacement string.
    Args:
        p: pattern
        r: replacement

    Returns:
        RegexRule
    """
    return RegexRule(re.compile(p), r)

def rename_key(key: str, rules: Iterable[RegexRule]) -> str:
    """
    Rename a single key using a series of regex rules.
    Args:
        key: str: The key to rename.
        rules: Iterable[RegexRule]: The regex rules to apply.

    Returns:
        str: The renamed key.
    """
    for rule in rules:
        key = rule.pattern.sub(rule.repl, key)
    return key

def rename_state_dict(state_dict: Dict[str, Tensor], rules: Iterable[RegexRule]) -> Dict[str, Tensor]:
    """
    Rename the keys of a state dict using a series of regex rules.
    Args:
        state_dict: Dict[str, Tensor]: The state dict to rename.
        rules: Iterable[RegexRule]: The regex rules to apply.

    Returns:
        Dict[str, Tensor]: The renamed state dict.
    """
    out: Dict[str, Tensor] = {}
    for k, v in state_dict.items():
        out[rename_key(k, rules)] = v
    return out

def transfer_data(model: nn.Module, mapped_state_dict: Dict[str, Tensor], verbose: bool = False) -> None:
    """
    Transfer weights from mapped_state_dict to the model, reporting any mismatches.
    Args:
        model: nn.Module: The model to load weights into.
        mapped_state_dict: Dict[str, Tensor]: The mapped state dict.
        verbose: bool: Whether to print a report of mismatches.

    Returns:

    """
    msd = model.state_dict()
    loadable: Dict[str, Tensor] = {}
    mism: List[Tuple[str, Tuple[int, ...], Optional[Tuple[int, ...]]]] = []
    for k, v in mapped_state_dict.items():
        if k in msd and msd[k].shape == v.shape:
            loadable[k] = v
        else:
            mism.append((k, tuple(v.shape), tuple(msd[k].shape) if k in msd else None))
    missing = sorted(set(msd.keys()) - set(loadable.keys()))
    unexpected = sorted(set(mapped_state_dict.keys()) - set(msd.keys()))
    if verbose:
        report = {
            "num_loadable": len(loadable),
            "missing_examples": missing[:15],
            "unexpected_examples": unexpected[:15],
            "mismatch_examples": mism[:10],
        }
        print(report)
    model.load_state_dict(loadable, strict=True)
