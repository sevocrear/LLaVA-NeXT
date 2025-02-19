import gradio as gr
from PIL import Image
import io
from model_eval_tools import LLMModel
import fitz

llm = LLMModel()


def process_document(file_obj):
    try:
        """Extracts text from a PDF file using fitz."""
        doc = fitz.open(file_obj)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return ""


# LLaVA‑NeXT inference function.
def llava_next_inference(prompt, image=None, video=None, document=None):
    combined_prompt = prompt

    images = []
    if image is not None:
        images.append(image)
    video_path = None
    if video is not None:
        video_path = video
        pass

    if document is not None:
        # Process the document to extract text.
        doc_text = process_document(document)
        combined_prompt += f" [Document content: {doc_text}]"

    # LLaVA‑NeXT model inference (e.g., model.chat(combined_prompt))
    output = llm.prompt(combined_prompt, images=images, video=video_path)
    response = f"{output}"
    return response


def chat_interface(text, image, video, document):
    return llava_next_inference(text, image, video, document)


with gr.Blocks(title="LLaVA‑NeXT Web Chat") as demo:
    gr.Markdown("# LLaVA‑NeXT Web Chat Interface")
    gr.Markdown(
        "Chat with the LLaVA‑NeXT model. Enter your message below and optionally upload an image, video, or document. "
    )
    gr.Markdown(
        "For videos and documents, ensure that you have appropriate pre‑processing (e.g., OCR for PDFs) in place."
    )
    gr.Markdown(
        "When use images, type `... in image /im`, same for videos. `/im` is changed to the image tokens inside. Example: `What is in the image /im?`. Without it, **you'll get unexpected results**"
    )

    with gr.Row():
        text_input = gr.Textbox(label="Your Message", placeholder="Type your message here...", lines=3)

    with gr.Row():
        image_input = gr.Image(label="Upload Image", type="numpy", optional=True)
        video_input = gr.Video(label="Upload Video", optional=True)
        document_input = gr.File(label="Upload Document", optional=True)
    response_output = gr.Textbox(label="Model Response")

    # When the text box is submitted or the Send button is clicked, process the inputs.
    text_input.submit(
        chat_interface, inputs=[text_input, image_input, video_input, document_input], outputs=response_output
    )
    gr.Button("Send").click(
        chat_interface, inputs=[text_input, image_input, video_input, document_input], outputs=response_output
    )

demo.launch()
