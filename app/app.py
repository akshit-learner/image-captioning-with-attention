"""
Gradio demo -- upload an image, get a generated caption plus a
visualization of what the model attended to for each word.

Run locally:      python app.py
Run on HF Spaces:  just push this file + requirements.txt + a checkpoint,
                    set app.py as the entry point.
"""

import tempfile
import gradio as gr

from predict import CaptionPredictor, plot_attention

predictor = None  # lazy-loaded so the UI opens instantly


def get_predictor():
    global predictor
    if predictor is None:
        predictor = CaptionPredictor()
    return predictor


def caption_image(image, method):
    p = get_predictor()

    # Gradio gives us a PIL image; save to a temp file since our pipeline
    # reads from disk (keeps preprocessing identical to training).
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.convert("RGB").save(tmp.name)
        image_path = tmp.name

        if method == "greedy (with attention map)":
            caption, attention_plots = p.generate_greedy(image_path)
            words = caption.split()
            fig = plot_attention(image_path, words, attention_plots)
            return caption, fig
        else:
            caption = p.generate_beam_search(image_path)
            return caption, None


with gr.Blocks(title="Image Captioning with Attention") as demo:
    gr.Markdown(
        "# Image Captioning with Visual Attention\n"
        "Upload an image and generate a caption using an attention-based "
        "encoder-decoder model trained on Flickr8k. Choose **beam search** "
        "for the best-quality caption, or **greedy** to also see an "
        "attention heatmap for each generated word."
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload an image")
            method_input = gr.Radio(
                ["beam search", "greedy (with attention map)"],
                value="beam search",
                label="Decoding method",
            )
            submit_btn = gr.Button("Generate caption", variant="primary")
        with gr.Column():
            caption_output = gr.Textbox(label="Generated caption")
            attention_output = gr.Plot(label="Attention map (greedy only)")

    submit_btn.click(
        fn=caption_image,
        inputs=[image_input, method_input],
        outputs=[caption_output, attention_output],
    )

if __name__ == "__main__":
    demo.launch()
