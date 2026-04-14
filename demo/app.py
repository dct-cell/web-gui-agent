"""Gradio demo for Web GUI Agent — live browser control with visualization."""
import argparse
from PIL import Image, ImageDraw

import gradio as gr

from model.config import InferenceConfig
from model.inference import WebGUIModel
from agent.browser import Browser
from agent.action_parser import format_action_text
from agent.prompt_builder import build_prompt


def draw_action_overlay(screenshot: Image.Image, action: dict, width: int, height: int) -> Image.Image:
    """Draw action indicator on screenshot (red dot for click, red box for type)."""
    img = screenshot.copy()
    draw = ImageDraw.Draw(img)

    if action["action"] in ("click", "type"):
        pos = action["position"]
        px, py = int(pos[0] * width), int(pos[1] * height)
        radius = 12
        draw.ellipse(
            [px - radius, py - radius, px + radius, py + radius],
            fill="red",
            outline="white",
            width=2,
        )
        if action["action"] == "type":
            text = action.get("value", "")
            draw.text((px + 15, py - 10), f'"{text}"', fill="red")

    return img


def create_demo(model: WebGUIModel, default_headless: bool = True):
    """Create and return the Gradio demo app."""

    def run_agent(url: str, instruction: str, max_steps: int):
        """Generator that yields (screenshot, log_text, status) after each step."""
        if not url.startswith("http"):
            url = "https://" + url

        browser = Browser(headless=default_headless)
        history = []
        log_lines = []

        try:
            browser.navigate(url)
            log_lines.append(f"Navigated to: {url}")
            log_lines.append(f"Task: {instruction}")
            log_lines.append("---")

            screenshot = browser.screenshot()
            yield screenshot, "\n".join(log_lines), f"Step 0/{max_steps} - Ready"

            for step_idx in range(max_steps):
                messages = build_prompt(screenshot, instruction, history)
                action = model.predict(messages)

                if action is None:
                    log_lines.append(f"Step {step_idx + 1}: Parse error, retrying...")
                    yield screenshot, "\n".join(log_lines), f"Step {step_idx + 1}/{max_steps} - Parse error"
                    continue

                action_text = format_action_text(action)
                log_lines.append(f"Step {step_idx + 1}: {action_text}")

                if action["action"] == "done":
                    answer = action.get("answer", "Task completed")
                    log_lines.append(f"\nDone: {answer}")
                    annotated = draw_action_overlay(screenshot, action, browser.width, browser.height)
                    yield annotated, "\n".join(log_lines), f"Completed in {step_idx + 1} steps"
                    return

                browser.execute(action)
                annotated = draw_action_overlay(screenshot, action, browser.width, browser.height)

                history.append({"screenshot": screenshot, "action": action})
                if len(history) > 2:
                    history.pop(0)

                screenshot = browser.screenshot()
                yield annotated, "\n".join(log_lines), f"Step {step_idx + 1}/{max_steps} - Running"

            log_lines.append(f"\nMax steps ({max_steps}) reached")
            yield screenshot, "\n".join(log_lines), f"Stopped - max {max_steps} steps"

        except Exception as e:
            log_lines.append(f"\nError: {e}")
            yield Image.new("RGB", (1344, 756), (50, 50, 50)), "\n".join(log_lines), f"Error: {e}"
        finally:
            browser.close()

    with gr.Blocks(title="Web GUI Agent Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Web GUI Agent Demo")
        gr.Markdown("Enter a URL and a task instruction. The agent will navigate the website automatically.")

        with gr.Row():
            url_input = gr.Textbox(label="URL", placeholder="https://en.wikipedia.org", scale=3)
            instruction_input = gr.Textbox(label="Task Instruction", placeholder="Search for Transformer and navigate to Architecture section", scale=4)
            max_steps_slider = gr.Slider(minimum=1, maximum=30, value=15, step=1, label="Max Steps", scale=1)

        run_btn = gr.Button("Run Agent", variant="primary")

        with gr.Row():
            screenshot_output = gr.Image(label="Browser Screenshot", type="pil", height=500)
            with gr.Column():
                status_output = gr.Textbox(label="Status", interactive=False)
                log_output = gr.Textbox(label="Execution Log", lines=15, interactive=False)

        run_btn.click(
            fn=run_agent,
            inputs=[url_input, instruction_input, max_steps_slider],
            outputs=[screenshot_output, log_output, status_output],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or HF model ID")
    parser.add_argument("--lora_path", type=str, default="", help="Path to LoRA weights")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    config = InferenceConfig(
        model_path=args.model_path,
        lora_path=args.lora_path,
        device=args.device,
    )
    model = WebGUIModel(config)
    demo = create_demo(model)
    demo.launch(server_port=args.port, share=args.share)
