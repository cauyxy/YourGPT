import os
import torch
from torch.utils.data import Dataset
import bitsandbytes as bnb

import gradio as gr

from peft import PeftModel, LoraConfig, TaskType, set_peft_model_state_dict, get_peft_model
from transformers import BloomForCausalLM, BloomTokenizerFast, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling


def _get_available_model():
    return os.listdir("./outputs/")


available_model = _get_available_model()
current_model = available_model[0]

PROMPT_F = ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:")
PROMPT_F_Train = ("Below is an instruction that describes a task. "
                  "Write a response that appropriately completes the request.\n\n"
                  "### Instruction:\n你叫什么名字？\n\n### Response:我的名字是{user_name}呀{eos}")

model_path = "bigscience/bloomz-560m"
tokenizer = BloomTokenizerFast.from_pretrained(model_path)
base_model = BloomForCausalLM.from_pretrained(model_path, return_dict=True, device_map='auto')

peft_model_path = f"./outputs/tutu/"
model = PeftModel.from_pretrained(base_model, peft_model_path)


class NamingDataset(Dataset):
    def __init__(self, user_name, tokenizer, max_length=2048):
        self.post_list = [PROMPT_F_Train.format_map({"user_name": user_name, "eos": tokenizer.eos_token})] * 100

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        txt = self.post_list[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }


def _update_model(lora_name):
    global model
    peft_model_path = f"./outputs/{lora_name}/adapter_model.bin"
    adapters_weights = torch.load(
        peft_model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_peft_model_state_dict(model, adapters_weights)


def _generate(instruction):
    model_input = PROMPT_F.format_map({"instruction": instruction})
    question = tokenizer(model_input, return_tensors='pt')
    tutu_output = model.generate(**question, max_new_tokens=50)
    model_output = tokenizer.decode(tutu_output[0], skip_special_tokens=True)
    return model_output.replace(model_input, "")


def generate(instruction, selected_model):
    global current_model
    if selected_model is None:
        return "Please select a model!"
    if current_model != selected_model:
        _update_model(selected_model)
        current_model = selected_model

    return _generate(instruction)


def training(model_name, user_name, progress=gr.Progress(track_tqdm=True)):
    if model_name in available_model:
        return f"{model_name} is occupied, please select another!", gr.update()
    progress(0, desc="Starting")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
    )
    model_to_train = BloomForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map='auto')
    model_to_train = get_peft_model(model_to_train, lora_config)
    train_dataset = NamingDataset(user_name, tokenizer, max_length=512)
    trainer = Trainer(
        model=model_to_train,
        train_dataset=train_dataset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=30,
            max_steps=58,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            output_dir='outputs'
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    model_to_train.save_pretrained(f"outputs/{model_name}/")
    del trainer
    del train_dataset
    del model_to_train
    torch.cuda.empty_cache()
    return f"{model_name} Training Progress Done!", gr.update(choices=_get_available_model())


def training_useful(path, progress=gr.Progress(track_tqdm=True)):
    return "Not Implanted"


with gr.Blocks() as demo:
    gr.Markdown("<center>Train a Model On Your Dataset</center>")
    with gr.Tab("Ask the Model"):
        ins_input = gr.Textbox("你叫什么名字？", label="Input")
        selected_model = gr.Radio(_get_available_model(), label="Model")
        model_output = gr.Textbox(label="Output")
        gen_button = gr.Button("Generate")
    with gr.Tab("Train A Name Model"):
        with gr.Row():
            model_name = gr.Text(placeholder="ikun", label="Model Name")
            name_input = gr.Text(placeholder="蔡徐坤", label="Your Name")
        train_output = gr.Text("Input your name and submit to train YourGPT", label="Training Progress")
        train_naming_bt = gr.Button("Start training")

    with gr.Tab("Train A Usefully Model"):
        userful_data_path = gr.Text(placeholder="./path/to/your/data", label="Data Path")
        userful_output = gr.Text("Input your name and submit to train YourGPT", label="Training Progress")
        train_useful_model = gr.Button("Start training")

    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at me...")

    gen_button.click(generate, inputs=[ins_input, selected_model], outputs=model_output)
    train_naming_bt.click(training, inputs=[model_name, name_input], outputs=[train_output, selected_model])
    train_useful_model.click(training, inputs=name_input, outputs=[train_output, selected_model])

if __name__ == "__main__":
    demo.queue(concurrency_count=10).launch()
