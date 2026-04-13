"""Gradio UI service - Main application."""

import gradio as gr
from shared.logging_config import setup_logging

logger = setup_logging("gradio-ui")


def build_interface() -> gr.Blocks:
    """Build the Gradio interface.

    TODO: Migrate from current app.py Gradio interface.
    Current implementation is a placeholder until migration.
    """
    with gr.Blocks(title="Local Media Processor") as interface:
        gr.Markdown("# Local Media Processor")
        gr.Markdown(
            "⚠️ **Migration in progress**: This UI will be connected to the API Gateway"
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload files for processing")
                file_input = gr.File(label="Upload File", file_count="multiple")
                process_btn = gr.Button("Process", variant="primary")

            with gr.Column():
                gr.Markdown("### Results")
                output = gr.Textbox(label="Status")

        # TODO: Connect to API gateway instead of direct processing
        process_btn.click(
            fn=lambda: "API Gateway integration not yet implemented",
            inputs=[],
            outputs=[output],
        )

    return interface


# Create Gradio app
app = build_interface()


if __name__ == "__main__":
    logger.info("Starting Gradio UI service on port 7860")
    app.launch(server_name="0.0.0.0", server_port=7860)
