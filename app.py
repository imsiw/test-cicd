import gradio as gr

def greet(name, intensity):
    return "Hello, World"

demo = gr.Interface(
    fn=greet,
    inputs=["image"],
    outputs=["text"],
    api_name="predict"
)

demo.launch()