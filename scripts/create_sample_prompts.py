#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT_TEMPLATES = [
    "Solve step by step. A train leaves city A at {a} km/h and another leaves city B at {b} km/h. They are {c} km apart and travel toward each other. When do they meet?",
    "Solve step by step. A shop discounts an item by {a}% and then adds {b}% tax. If the original price is {c} euros, what is the final price?",
    "Solve step by step. A rectangle has perimeter {a} and one side is {b}. What is its area?",
    "Solve step by step. A recipe uses {a} grams of flour for {b} servings. How much flour is needed for {c} servings?",
    "Solve step by step. A number is multiplied by {a}, then {b} is added, giving {c}. What was the original number?",
    "Solve step by step. A tank is filled by one pipe in {a} hours and another in {b} hours. How long do they take together?",
    "Solve step by step. A class has {a} students. {b}% are absent today. How many students are present?",
    "Solve step by step. A sequence starts at {a} and increases by {b} each term. What is the {c}th term?",
    "Solve step by step. A cyclist travels {a} km in {b} hours, then {c} km in 1 hour. What is the average speed?",
    "Solve step by step. A bag contains {a} red balls, {b} blue balls, and {c} green balls. What is the probability of drawing a blue ball?",
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Create a simple JSONL prompt set for MGT-B experiments.")
    parser.add_argument("--output", default="data/prompts.jsonl")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for idx in range(args.count):
            template = PROMPT_TEMPLATES[idx % len(PROMPT_TEMPLATES)]
            a = 12 + (idx * 7) % 89
            b = 3 + (idx * 5) % 47
            c = 20 + (idx * 11) % 180
            row = {
                "id": f"sample-{idx:04d}",
                "prompt": template.format(a=a, b=b, c=c),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {args.count} prompts to {output}")


if __name__ == "__main__":
    main()
