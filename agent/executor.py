"""Task executor — runs the screenshot→predict→act loop."""
from dataclasses import dataclass, field
from PIL import Image

from agent.prompt_builder import build_prompt


@dataclass
class ExecutionResult:
    success: bool
    answer: str = ""
    steps_taken: int = 0
    step_history: list[dict] = field(default_factory=list)


class TaskExecutor:
    def __init__(self, model, browser, max_steps: int = 15):
        self.model = model
        self.browser = browser
        self.max_steps = max_steps

    def run(self, url: str, instruction: str) -> ExecutionResult:
        self.browser.navigate(url)
        history = []
        step_history = []

        for step_idx in range(self.max_steps):
            screenshot = self.browser.screenshot()
            messages = build_prompt(screenshot, instruction, history)
            action = self.model.predict(messages)

            if action is None:
                step_history.append({
                    "step": step_idx + 1,
                    "screenshot": screenshot,
                    "action": {"action": "parse_error"},
                })
                continue

            step_record = {
                "step": step_idx + 1,
                "screenshot": screenshot,
                "action": action,
            }
            step_history.append(step_record)

            if action["action"] == "done":
                return ExecutionResult(
                    success=True,
                    answer=action.get("answer", "Task completed"),
                    steps_taken=step_idx + 1,
                    step_history=step_history,
                )

            self.browser.execute(action)

            history.append({"screenshot": screenshot, "action": action})
            if len(history) > 2:
                history.pop(0)

        return ExecutionResult(
            success=False,
            answer="Max steps reached",
            steps_taken=self.max_steps,
            step_history=step_history,
        )
