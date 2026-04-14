"""Online end-to-end evaluation with live browser tasks.

Runs the agent on predefined tasks and reports success rate.
Tasks require human judgment for success verification.
"""
import json
import argparse
from datetime import datetime
from pathlib import Path

from model.config import InferenceConfig
from model.inference import WebGUIModel
from agent.browser import Browser
from agent.executor import TaskExecutor


EVAL_TASKS = [
    {
        "id": "wiki_search",
        "url": "https://en.wikipedia.org",
        "instruction": "Search for 'Transformer (deep learning)' and navigate to the Architecture section",
        "description": "Wikipedia search and section navigation",
    },
    {
        "id": "google_search",
        "url": "https://www.google.com",
        "instruction": "Search for 'weather today'",
        "description": "Simple Google search",
    },
    {
        "id": "example_links",
        "url": "https://example.com",
        "instruction": "Click on the 'More information...' link",
        "description": "Simple link click on example.com",
    },
    {
        "id": "wiki_nav",
        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "instruction": "Navigate to the History section of this article",
        "description": "Navigate within a Wikipedia article",
    },
    {
        "id": "baidu_search",
        "url": "https://www.baidu.com",
        "instruction": "搜索 '天气预报' 并点击搜索按钮",
        "description": "Baidu search in Chinese",
    },
]


def run_online_eval(
    model: WebGUIModel,
    tasks: list[dict] = None,
    max_steps: int = 15,
    headless: bool = True,
    output_dir: str = "./eval_results",
) -> list[dict]:
    if tasks is None:
        tasks = EVAL_TASKS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    for task in tasks:
        print(f"\n--- Running: {task['description']} ---")
        browser = Browser(headless=headless)
        executor = TaskExecutor(model=model, browser=browser, max_steps=max_steps)

        try:
            result = executor.run(task["url"], task["instruction"])

            task_dir = output_path / task["id"]
            task_dir.mkdir(exist_ok=True)
            for step in result.step_history:
                img = step.get("screenshot")
                if img:
                    img.save(task_dir / f"step_{step['step']:02d}.png")

            task_result = {
                "task_id": task["id"],
                "description": task["description"],
                "instruction": task["instruction"],
                "agent_success": result.success,
                "steps_taken": result.steps_taken,
                "answer": result.answer,
                "human_verified": None,
            }
            results.append(task_result)

            status = "DONE" if result.success else f"STOPPED ({result.answer})"
            print(f"  Result: {status} in {result.steps_taken} steps")

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "task_id": task["id"],
                "description": task["description"],
                "error": str(e),
                "human_verified": None,
            })
        finally:
            browser.close()

    results_file = output_path / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {results_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = InferenceConfig(
        model_path=args.model_path,
        lora_path=args.lora_path,
        device=args.device,
    )
    model = WebGUIModel(config)
    run_online_eval(model, max_steps=args.max_steps, headless=args.headless, output_dir=args.output_dir)
