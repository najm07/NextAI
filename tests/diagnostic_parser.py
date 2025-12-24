"""Diagnostic script for instruction parser accuracy."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agent.language_agent import LanguageAgent, LanguageAgentConfig


def run_diagnostic():
    """Run instruction parsing diagnostic."""
    config = LanguageAgentConfig()
    agent = LanguageAgent(config)
    
    # Test cases: (instruction, expected_type, expected_args)
    test_cases = [
        ("increase object 0 attr0", 0, {"i": 0, "a": 0}),
        ("decrease object 1 attr2", 1, {"i": 1, "a": 2}),
        ("increase object 2", 0, {"i": 2}),
        ("make object 0 larger", 0, {"i": 0}),
        ("swap object 0 and object 1", 6, {"i": 0, "j": 1}),
        ("swap object 2 with object 3", 6, {"i": 2, "j": 3}),
        ("toggle relation 0 between object 1 and object 2", 3, {"r": 0, "i": 1, "j": 2}),
        ("set relation 1 between object 0 and object 3 on", 4, {"r": 1, "i": 0, "j": 3}),
        ("connect object 0 to object 1", 4, {"i": 0, "j": 1}),
        ("disconnect object 2 from object 3", 5, {"i": 2, "j": 3}),
        ("noise object 2", 7, {"i": 2}),
        ("add noise to object 1", 7, {"i": 1}),
    ]
    
    INT_NAMES = [
        "IncreaseAttr", "DecreaseAttr", "SetAttrToward",
        "ToggleRel", "SetRelOn", "SetRelOff",
        "SwapObjects", "NoiseBurstAttr"
    ]
    
    print("Instruction Parsing Diagnostic")
    print("=" * 70)
    
    correct_type = 0
    correct_i = 0
    correct_j = 0
    
    for inst, expected_type, expected_args in test_cases:
        u_type, args, conf = agent.parse_instruction(inst)
        
        type_ok = u_type == expected_type
        i_ok = args.get("i") == expected_args.get("i", args.get("i"))
        j_ok = args.get("j") == expected_args.get("j", args.get("j"))
        
        if type_ok:
            correct_type += 1
        if i_ok:
            correct_i += 1
        if j_ok:
            correct_j += 1
        
        all_ok = type_ok and i_ok and (expected_args.get("j") is None or j_ok)
        status = "[OK]  " if all_ok else "[FAIL]"
        
        print(f'{status} "{inst}"')
        print(f'       Got:      {INT_NAMES[u_type]:15s} i={args.get("i")}, j={args.get("j")}')
        print(f'       Expected: {INT_NAMES[expected_type]:15s} i={expected_args.get("i")}, j={expected_args.get("j")}')
        print()
    
    n = len(test_cases)
    print("=" * 70)
    print(f"Type accuracy: {correct_type}/{n} = {100*correct_type/n:.0f}%")
    print(f"Arg i accuracy: {correct_i}/{n} = {100*correct_i/n:.0f}%")
    print(f"Arg j accuracy: {correct_j}/{n} = {100*correct_j/n:.0f}%")
    
    return correct_type / n, correct_i / n


if __name__ == "__main__":
    run_diagnostic()

