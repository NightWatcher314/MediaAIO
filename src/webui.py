from time import sleep
import gradio as gr
import whisper_page
import final2x_page


def whisper(text):
    sleep(10)
    return text.lower() + "..."


demo = gr.TabbedInterface(
    [final2x_page.ui(), whisper_page.ui()], ["Whisper", "Super_Resolution"]
)
demo.queue(max_size=4)

if __name__ == "__main__":
    demo.launch(max_threads=4, share=False, debug=True, inline=True, auth=None)
