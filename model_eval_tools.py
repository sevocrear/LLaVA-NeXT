import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import cv2
import copy
import requests
from PIL import Image, ImageFile
from decord import VideoReader, cpu

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

class LLMModel():
    def __init__(self):
        # Load model configuration
        pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
        model_name = "llava_qwen"
        self.device = "cuda"
        device_map = "auto"
        llava_model_args = {"multimodal": True}
        overwrite_config = {"image_aspect_ratio": "pad"}
        llava_model_args["overwrite_config"] = overwrite_config

        # Load tokenizer, model, image processor and max_length from the pretrained model
        self.tokenizer, model, self.image_processor, self.max_length = load_pretrained_model(
            pretrained, None, model_name, device_map=device_map, **llava_model_args
        )
        model.eval()
        self.model = model

        # Initialize conversation history using a conversation template.
        # This conversation instance will accumulate the dialog turns.
        self.conv = copy.deepcopy(conv_templates["qwen_1_5"])

        self.video_tensors = []
        self.image_tensors = []
        self.images_list = []
        self.video_frames = []
        self.video_tensors = []
    def clean_history(self):
        # Initialize conversation history using a conversation template.
        # This conversation instance will accumulate the dialog turns.
        self.conv = copy.deepcopy(conv_templates["qwen_1_5"])

        self.video_tensors = []
        self.image_tensors = []
        self.images_list = []
        self.video_frames = []
        self.video_tensors = []
    # Function to extract uniformly sampled frames from a video
    def load_video(self, video_path, max_frames_num):
        if isinstance(video_path, str):
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames

    # Prepares images from a list of inputs (URLs, file paths, or PIL Images)
    def prepare_images(self, images_in: list = ["https://raw.githubusercontent.com/haotian-liu/LLaVA/main/images/llava_logo.png"]):
        if images_in:
            images = []
            for image in images_in:
                if isinstance(image, Image.Image):
                    pass
                elif isinstance(image, (np.ndarray, np.generic)):
                    cv2.imwrite("image_test.png", image)
                    image = Image.fromarray(image)
                elif isinstance(image, str) and "http" in image:
                    image = Image.open(requests.get(image, stream=True).raw)
                else:
                    image = Image.open(image)
                images.append(image)
            image_tensors = process_images(images, self.image_processor, self.model.config)
            image_tensors = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors]
            return image_tensors, images
        dummy = torch.zeros((3, 384, 384)).half().to(self.device)
        return [dummy], [dummy]

    # Prepares video frames as image tensors
    def prepare_video(self, video_path="jobs.mp4"):
        if video_path is not None:
            video_frames = self.load_video(video_path, 16)
            frames_tensor = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().to(self.device)
            # Wrap the frames tensor in a list to match the expected input format
            return [frames_tensor], frames_tensor
        dummy = torch.zeros((3, 384, 384)).half().to(self.device)
        return [dummy], [dummy]

    # This function updates the conversation history with the new user input and returns a model response.
    def prompt(self, text: str = "Hello", images=None, video=None):
        # Process user text input
        user_message = text.replace("/im", DEFAULT_IMAGE_TOKEN)

        # Append the user message to the conversation history
        self.conv.append_message(self.conv.roles[0], user_message)
        # Append a placeholder for the model's answer
        self.conv.append_message(self.conv.roles[1], None)

        # Get the full conversation prompt including history
        prompt_question = self.conv.get_prompt()

        # Prepare image and video tensors
        image_tensors, images_list = self.prepare_images(images)
        video_tensors, video_frames = self.prepare_video(video)
        self.image_tensors.extend(image_tensors)
        self.images_list.extend(images_list)
        self.video_frames.extend(video_frames)
        self.video_tensors.extend(video_tensors)

        # Generate a response using the updated conversation prompt
        response_text = self._generate_outputs(prompt_question, self.images_list, self.image_tensors, self.video_frames, self.video_tensors)
        # Update the conversation history with the model's response
        self.conv.messages[-1][-1] = response_text
        return response_text

    # Generates outputs using the model's generate() function.
    def _generate_outputs(self, prompt_question, images, image_tensors, video_frames, video_tensors, temp=0):
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_sizes = [image.size for image in images]
        video_image_sizes = [frame.size for frame in video_frames]
        cont = self.model.generate(
            input_ids,
            images=image_tensors + video_tensors,
            image_sizes=image_sizes + video_image_sizes,
            do_sample=False,
            temperature=temp,
            max_new_tokens=4096,
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id),
            use_cache=False
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs[0]

if __name__ == "__main__":
    llm = LLMModel()

    # Simple command-line interaction that preserves conversation history.
    while True:
        prompt_input = input("You: ")
        if prompt_input.lower() in ["exit", "quit"]:
            break
        output = llm.prompt(prompt_input, images=["image.png"])  # Replace or add video/file inputs as needed.
        print("LLava-Next:", output)
