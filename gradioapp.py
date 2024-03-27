import gradio as gr
from huggingface_hub import InferenceClient

def echo(message, history):
    return message

client = InferenceClient(model="http://34.208.241.171:8080")

def inference(message, history):
    partial_message = ""
    for token in client.text_generation(message, max_new_tokens=20, stream=True):
        partial_message += token
        yield partial_message

demo = gr.ChatInterface(fn=inference, examples=["hello", "hola", "merhaba"], title="Mistral Bot")

demo.launch()
