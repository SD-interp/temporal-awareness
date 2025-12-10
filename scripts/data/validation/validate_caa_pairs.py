import ask_llm_council
import caa_constants
import json
from pathlib import Path
import os
import argparse
import time

def get_parser():
    parser = argparse.ArgumentParser()
    root_dir = Path(__file__).parent / ".." / ".." / ".."
    implicit_dataset = str(root_dir / "data" / "raw" / "temporal_scope_implicit.json")
    parser.add_argument("--dataset", type=str, default=implicit_dataset)
    parser.add_argument("--agent", choices=['council', 'claude'], default='council')
    return parser

def validate_dataset(dataset_path, agent):
    pairs = None
    with open(dataset_path) as f:
        json_content = json.load(f)
        assert 'pairs' in json_content
        pairs = json_content['pairs']

    results = []
    ask_agent = None
    if agent == "council":
        ask_agent = ask_llm_council.ask_llm_council
        get_agent_info = ask_llm_council.get_council
    elif agent == "claude":
        raise Exception("Claude is not fully supported yet")
        # ask_agent = ask_claude_sonnet.ask_claude_sonnet
    else:
        raise("Shouldn't reach that point")

    no_explicit_temporal_words_rule = caa_constants.no_explicit_temporal_words_rule \
        if 'implicit' in dataset_path else caa_constants.allowed_explicit_temporal_words_rule
    print(f"No temporal words rule for given dataset: {no_explicit_temporal_words_rule}")

    for i, pair in enumerate(pairs):
        response = ask_agent(
            caa_constants.validation_prompt.format(
                no_explicit_temporal_words_rule=no_explicit_temporal_words_rule,
                question=pair['question'],
                option_a=pair['immediate'],
                option_b=pair['long_term'],
                category=pair.get('category', 'unknown')) + \
            caa_constants.validation_prompt_return_hint,
            caa_constants.validation_response_format)
        result = json.loads(response)

        result['pair_index'] = i
        result["pair"] = pair
        print(f"Pair {i}:")
        print(result)
      
        results.append(result)
        
        avg = result['average']
        status = "✓" if float(avg) >= 3.5 else ("⚠" if float(avg) >= 2.5 else "✗")
        print(f"{status} Pair {i}: {avg:.1f}/5 - {result.get('issues', [])}")
    
    # Summary
    avgs = [r['average'] for r in results]
    print(f"\n=== SUMMARY ===")
    print(f"Mean: {sum(avgs)/len(avgs):.2f}/5")
    print(f"Excellent (4.5+): {sum(1 for a in avgs if a >= 4.5)}")
    print(f"Good (3.5-4.4): {sum(1 for a in avgs if 3.5 <= a < 4.5)}")
    print(f"Marginal (2.5-3.4): {sum(1 for a in avgs if 2.5 <= a < 3.5)}")
    print(f"Poor (<2.5): {sum(1 for a in avgs if a < 2.5)}")
    summary = {
        "Mean" : sum(avgs)/len(avgs),
        "Excellent (4.5+)" : sum(1 for a in avgs if a >= 4.5),
        "Good (3.5-4.4)" : sum(1 for a in avgs if 3.5 <= a < 4.5),
        "Marginal (2.5-3.4)" : sum(1 for a in avgs if 2.5 <= a < 3.5),
        "Poor (<2.5)" : sum(1 for a in avgs if a < 2.5),
        "Assessed by" : get_agent_info()
    }
    result_dict = {"summary" : summary, "results" : results}
    return result_dict

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    results = validate_dataset(args.dataset, args.agent)
    dataset_name = Path(args.dataset).stem
    result_file_name = f"validation_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.json"
    results_dir = Path(__file__).parent / "results"
    with open(results_dir / result_file_name, "w") as f:
        json.dump(results, f, indent=2)
