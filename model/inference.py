"""Model inference — loads Qwen2-VL with optional LoRA and generates action predictions."""
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

from model.config import InferenceConfig
from agent.action_parser import parse_action


class WebGUIModel:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            config.model_path,
            trust_remote_code=True,
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model_path,
            torch_dtype="auto",
            device_map=config.device,
            trust_remote_code=True,
        )

        if config.lora_path:
            self.model = PeftModel.from_pretrained(self.model, config.lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()

    def predict(self, messages: list[dict]) -> dict | None:
        """Run inference on a Qwen2-VL message list and return parsed action.

        Args:
            messages: Qwen2-VL format messages with image and text content.

        Returns:
            Parsed action dict or None if parsing fails.
        """
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._extract_visual_inputs(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
        )
        # Decode only generated tokens
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return parse_action(output_text)

    def _extract_visual_inputs(self, messages: list[dict]):
        """Extract PIL images from messages content."""
        images = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        img = part.get("image")
                        if isinstance(img, Image.Image):
                            images.append(img)
        return images if images else None, None
