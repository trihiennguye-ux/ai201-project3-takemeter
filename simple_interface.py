import ipywidgets as widgets
from IPython.display import display, HTML
from datasets import Dataset
import torch
import numpy as np

def classify_new_post(post_text):
    if not post_text.strip():
        return "Please enter some text to classify."

    # Create a HuggingFace Dataset for the single input text
    input_df = pd.DataFrame([{"text": post_text}])
    input_dataset = Dataset.from_pandas(
        input_df.rename(columns={"text": "text"})
    )
    tokenized_input_dataset = input_dataset.map(tokenize, batched=True)

    # Run inference using the fine-tuned trainer
    output = trainer.predict(tokenized_input_dataset)
    pred_ids = np.argmax(output.predictions, axis=-1)
    probs = torch.nn.functional.softmax(
        torch.tensor(output.predictions), dim=-1
    ).numpy()

    predicted_label_id = pred_ids[0]
    predicted_label_name = ID_TO_LABEL[predicted_label_id]
    confidence = probs[0][predicted_label_id]

    return f"**Predicted Label:** {predicted_label_name}<br>\n**Confidence:** {confidence:.2f}"

# Create widgets
text_input = widgets.Textarea(
    placeholder='Enter your post here...', 
    description='Post:', 
    layout=widgets.Layout(width='auto', height='100px')
)
classify_button = widgets.Button(description='Classify Post')
output_area = widgets.Output()

def on_classify_button_clicked(b):
    with output_area:
        output_area.clear_output()
        result = classify_new_post(text_input.value)
        display(HTML(result))

classify_button.on_click(on_classify_button_clicked)

display(text_input, classify_button, output_area)
