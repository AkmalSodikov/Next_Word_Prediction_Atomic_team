import math
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from tqdm import tqdm
import sys, signal


BASE_MODEL_NAME = "distilgpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8
EPOCHS = 3
MAX_LENGTH = 512
SAVE_NAME = "lora_bst"



tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def save_on_interrupt(path="lora_interrupted"):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved to: {path}")
    sys.exit(0)

def handler(sig, frame):
    save_on_interrupt()

signal.signal(signal.SIGINT, handler)



def prepare_dataset(split="train"):
    ds = load_dataset("blended_skill_talk", split=split)
    data = []

    for ex in ds:
        prev = ex.get("previous_utterance", [])
        free = ex.get("free_messages", [])

        if not prev or not free:
            continue

        text = "\n".join(prev) + "\n" + free[0]
        if len(text) > 30:
            data.append(text)

    return data


def collate_fn(batch):
    return tokenizer(
        batch,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )


def train_epoch(model, dataloader, optimizer, epoch):
    model.train()
    total_loss = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)

        labels = ids.clone()
        labels[mask == 0] = -100

        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_loss = loss.item()
        batch_ppl = math.exp(batch_loss) if batch_loss < 20 else float("inf")

        progress.set_postfix({"loss": f"{batch_loss:.4f}", "ppl": f"{batch_ppl:.2f}"})

        total_loss += batch_loss * ids.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    avg_ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    return avg_loss, avg_ppl


def cli_predict(model):
    model.eval()
    print("\nGPT2-LoRA Next Word Predictor")
    print("Type a sentence. Type 'exit' to quit.\n")

    while True:
        text = input("You: ").strip()
        if text.lower() in {"exit", "quit"}:
            break

        enc = tokenizer(text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            logits = model(**enc).logits

        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)

        entropy = -torch.sum(probs * torch.log(probs + 1e-12))
        ppl = torch.exp(entropy).item()

        print(f"\nPerplexity: {ppl:.3f}\nTop predictions:")

        top_probs, top_ids = torch.topk(probs, 5)
        for i, (p, idx) in enumerate(zip(top_probs, top_ids), 1):
            token = tokenizer.decode([idx]).strip()
            print(f"{i}. {token}  (prob={p.item():.4f})")
        print()


def save_plots(losses, ppls):

    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker="o")
    plt.title("LoRA Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("lora_loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(ppls, marker="o", color="orange")
    plt.title("LoRA Training Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.grid(True)
    plt.savefig("lora_ppl_curve.png")
    plt.close()

    print("Saved plots: lora_loss_curve.png, lora_ppl_curve.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "cli"], required=True)
    parser.add_argument("--lora_path", default=f"{SAVE_NAME}_best")
    args = parser.parse_args()

    if args.mode == "train":
        print("\n🔧 Training mode")

        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
        base_model.config.pad_token_id = tokenizer.pad_token_id

        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn"],
        )

        model = get_peft_model(base_model, lora_cfg).to(DEVICE)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        train_data = prepare_dataset("train")
        loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=collate_fn)

        epoch_losses = []
        epoch_ppls = []

        best_loss = float("inf")

        for epoch in range(1, EPOCHS + 1):
            avg_loss, avg_ppl = train_epoch(model, loader, optimizer, epoch)

            epoch_losses.append(avg_loss)
            epoch_ppls.append(avg_ppl)

            print(f"Epoch {epoch} → Loss={avg_loss:.4f}, PPL={avg_ppl:.2f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save_pretrained(f"{SAVE_NAME}_best")
                tokenizer.save_pretrained(f"{SAVE_NAME}_best")
                print("✓ Saved best checkpoint")

        save_plots(epoch_losses, epoch_ppls)

        return

    else:
        print(f"\nLoading LoRA weights from: {args.lora_path}")
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
        model = PeftModel.from_pretrained(base, args.lora_path).to(DEVICE)
        model.eval()

        cli_predict(model) 


if __name__ == "__main__":
    main()
