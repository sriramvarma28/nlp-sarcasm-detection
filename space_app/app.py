"""
BESSTIE Sarcasm Classifier — HuggingFace Space
COMM061 NLP Coursework, Group PG19
"""
import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

VARIETIES = ["en-UK", "en-AU", "en-IN"]
LABEL_NAMES = ["Not Sarcastic", "Sarcastic"]

ROBERTA_REPOS = {
    "en-UK": "Sriram28/besstie-roberta-en-uk",
    "en-AU": "Sriram28/besstie-roberta-en-au",
    "en-IN": "Sriram28/besstie-roberta-en-in",
}

LORA_REPOS = {
    "en-UK": "Sriram28/besstie-lora-en-uk",
    "en-AU": "Sriram28/besstie-lora-en-au",
    "en-IN": "Sriram28/besstie-lora-en-in",
}

TINYLLAMA_BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ROBERTA_BASE = "roberta-base"

print("Loading models — this takes ~2 minutes on CPU startup...")

# RoBERTa: 3 models, fp32 on CPU
rb_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_BASE)
roberta_models = {}
for v in VARIETIES:
    print(f"  Loading RoBERTa {v}...")
    roberta_models[v] = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_REPOS[v], torch_dtype=torch.float32
    ).eval()

# TinyLlama base + 3 LoRA adapters
print("  Loading TinyLlama base...")
tl_tokenizer = AutoTokenizer.from_pretrained(TINYLLAMA_BASE)
if tl_tokenizer.pad_token is None:
    tl_tokenizer.pad_token = tl_tokenizer.eos_token
    tl_tokenizer.pad_token_id = tl_tokenizer.eos_token_id

tl_base = AutoModelForSequenceClassification.from_pretrained(
    TINYLLAMA_BASE, num_labels=2, torch_dtype=torch.float32,
)
tl_base.config.pad_token_id = tl_tokenizer.pad_token_id

print("  Loading LoRA adapters...")
lora_model = PeftModel.from_pretrained(tl_base, LORA_REPOS["en-UK"], adapter_name="en-UK")
for v in ["en-AU", "en-IN"]:
    lora_model.load_adapter(LORA_REPOS[v], adapter_name=v)
lora_model.eval()

print("All models loaded — ready to serve.")


@torch.no_grad()
def predict_roberta(text, variety):
    inputs = rb_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    logits = roberta_models[variety](**inputs).logits
    probs = F.softmax(logits, dim=-1)[0].numpy()
    pred = int(np.argmax(probs))
    return LABEL_NAMES[pred], float(probs[pred]), float(probs[0]), float(probs[1])


@torch.no_grad()
def predict_lora(text, variety):
    lora_model.set_adapter(variety)
    inputs = tl_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    logits = lora_model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0].numpy()
    pred = int(np.argmax(probs))
    return LABEL_NAMES[pred], float(probs[pred]), float(probs[0]), float(probs[1])


def format_card(model_name, variety, label, conf, p0, p1):
    if label == "—":
        return f"### {model_name}\n*Awaiting input...*"
    emoji = "🎭" if label == "Sarcastic" else "💬"
    color = "#ef4444" if label == "Sarcastic" else "#10b981"
    return (
        f"### {model_name}\n"
        f"**Variety:** {variety}\n\n"
        f"<div style='font-size: 1.5em; color: {color}; font-weight: bold;'>{emoji} {label}</div>\n\n"
        f"**Confidence:** {conf:.1%}\n\n"
        f"- Not Sarcastic: {p0:.1%}\n"
        f"- Sarcastic: {p1:.1%}"
    )


def classify(text, variety):
    if not text or not text.strip():
        empty = ("—", 0.0, 0.0, 0.0)
        return (
            format_card("RoBERTa-base", variety, *empty),
            format_card("TinyLlama + LoRA", variety, *empty),
            {}, {},
        )
    rb_label, rb_conf, rb_p0, rb_p1 = predict_roberta(text, variety)
    lo_label, lo_conf, lo_p0, lo_p1 = predict_lora(text, variety)
    rb_card = format_card("RoBERTa-base (125M)", variety, rb_label, rb_conf, rb_p0, rb_p1)
    lo_card = format_card("TinyLlama-1.1B + LoRA", variety, lo_label, lo_conf, lo_p0, lo_p1)
    rb_probs = {LABEL_NAMES[0]: rb_p0, LABEL_NAMES[1]: rb_p1}
    lo_probs = {LABEL_NAMES[0]: lo_p0, LABEL_NAMES[1]: lo_p1}
    return rb_card, lo_card, rb_probs, lo_probs


EXAMPLES = [
    ["Just love to see rich families getting a free home renovation.", "en-AU"],
    ["The Royal Mail are absolutely amazing at delivering parcels on time.", "en-UK"],
    ["His condition after spending some quality time with police", "en-IN"],
    ["This product is well-made and arrived quickly. Highly recommend.", "en-UK"],
    ["Beautiful weather we're having - perfect for the picnic I cancelled.", "en-AU"],
]

with gr.Blocks(title="BESSTIE Sarcasm Classifier") as app:
    gr.Markdown(
        "# 🎭 BESSTIE Sarcasm Classifier\n"
        "**COMM061 NLP Coursework — Group PG19**\n\n"
        "Compare two fine-tuned models on sarcasm detection across English varieties:\n"
        "- **RoBERTa-base** (125M params, full fine-tuning)\n"
        "- **TinyLlama-1.1B + LoRA adapters** (1.1B params, parameter-efficient)\n\n"
        "Each model has 3 variety-specific versions (en-UK, en-AU, en-IN). "
        "Pick a variety to use that variety's specialized model.\n\n"
        "⚠️ *Running on CPU — first prediction takes ~30s, subsequent ones ~10s.*"
    )
    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(label="Input text", placeholder="Enter text to classify...", lines=3)
        with gr.Column(scale=1):
            variety_input = gr.Radio(choices=VARIETIES, value="en-AU",
                                     label="English variety",
                                     info="Selects which variety-specific model to use")
    classify_btn = gr.Button("Classify", variant="primary", size="lg")
    with gr.Row():
        with gr.Column():
            rb_output = gr.Markdown(label="RoBERTa")
            rb_probs_out = gr.Label(num_top_classes=2, label="RoBERTa probabilities")
        with gr.Column():
            lo_output = gr.Markdown(label="LoRA")
            lo_probs_out = gr.Label(num_top_classes=2, label="LoRA probabilities")
    gr.Markdown("### Try one of these examples")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[text_input, variety_input],
        outputs=[rb_output, lo_output, rb_probs_out, lo_probs_out],
        fn=classify, cache_examples=False,
    )
    classify_btn.click(
        fn=classify,
        inputs=[text_input, variety_input],
        outputs=[rb_output, lo_output, rb_probs_out, lo_probs_out],
    )
    gr.Markdown(
        "---\n"
        "*Models trained on the BESSTIE-CW-26 dataset. "
        "Best diagonal: RoBERTa en-AU → en-AU, Macro-F1 = 0.778.*"
    )

app.launch()
