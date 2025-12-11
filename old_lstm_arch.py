import os
import re
import pickle
import readline
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from datasets import load_dataset

from ai_core.model_definitions import Vocabulary, NextWordLSTM, TextDataset


def train_one_epoch(model, train_loader, criterion, optimizer, device):

    model.train()
    batch_losses = []
    batch_accs = []

    total_loss = 0
    total_correct = 0
    total_preds = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_item = loss.item()
        batch_losses.append(loss_item)

        predicted = outputs.argmax(dim=-1)

        mask = (targets != 0).float()
        correct = ((predicted == targets) * mask).sum().item()
        total = mask.sum().item()

        acc = correct / total if total > 0 else 0.0
        batch_accs.append(acc)

        total_loss += loss_item
        total_correct += correct
        total_preds += total

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss={loss_item:.4f}, Acc={acc:.4f}")

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = total_correct / total_preds if total_preds > 0 else 0.0

    return batch_losses, batch_accs, epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):

    model.eval()
    total_loss = 0
    total_correct = 0
    total_preds = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()

            predicted = outputs.argmax(dim=-1)
            mask = (targets != 0).float()

            correct = ((predicted == targets) * mask).sum().item()
            total = mask.sum().item()

            total_correct += correct
            total_preds += total

    avg_loss = total_loss / len(val_loader)
    avg_acc = total_correct / total_preds if total_preds > 0 else 0.0

    return avg_loss, avg_acc


def predict_next_words(model, vocab, text, device, top_k=5):
    model.eval()
    tokens = vocab.encode(text)

    if len(tokens) == 0:
        return []

    input_tensor = torch.tensor([tokens]).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        last_output = output[0, -1, :]
        probs = torch.softmax(last_output, dim=0)
        top_probs, top_indices = torch.topk(probs, top_k)

    predictions = []
    for idx, prob in zip(top_indices.cpu().numpy(), top_probs.cpu().numpy()):
        word = vocab.idx2word.get(idx, "<UNK>")
        if word not in ["<PAD>", "<UNK>", "<START>", "<END>"]:
            predictions.append((word, prob))

    return predictions




class WordCompleter:
    def __init__(self, model, vocab, device):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.matches = []

    def complete(self, text, state):
        if state == 0:
            line = readline.get_line_buffer()
            preds = predict_next_words(self.model, self.vocab, line, self.device, top_k=10)
            self.matches = [word for word, _ in preds]

        try:
            return self.matches[state]
        except IndexError:
            return None


def interactive_cli(model, vocab, device):
    completer = WordCompleter(model, vocab, device)
    readline.set_completer(completer.complete)
    readline.parse_and_bind("tab: complete")
    readline.set_completer_delims(" \t\n")

    print("\nLSTM NEXT WORD PREDICTOR CLI")
    print("Press TAB to autocomplete. Type 'exit' to quit.\n")

    while True:
        try:
            text = input("You: ").strip()

            if text.lower() in ("quit", "exit"):
                break

            preds = predict_next_words(model, vocab, text, device)
            print("\nTop predictions:")
            for i, (w, p) in enumerate(preds, 1):
                print(f" {i}. {w} ({p:.4f})")
            print()

        except (KeyboardInterrupt, EOFError):
            break



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    model_path = "next_word_lstm.pt"
    vocab_path = "vocab.pkl"

    if os.path.exists(model_path):
        vocab = pickle.load(open(vocab_path, "rb"))
        model = NextWordLSTM(len(vocab.word2idx)).to(device)
        model.load_state_dict(torch.load(model_path))
        print("\nLoaded trained model. Starting CLI...\n")
        return interactive_cli(model, vocab, device)


    print("\nLoading dataset...")
    ds = load_dataset("blended_skill_talk", split="train")

    data = []
    for ex in ds:
        prev = ex.get("previous_utterance", [])
        free = ex.get("free_messages", [])
        if prev and free:
            text = "\n".join(prev) + "\n" + free[0]
            if len(text) > 30:
                data.append(text)

    print("Building vocabulary...")
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocab(data)

    print("Vocab size:", len(vocab.word2idx))


    sequences = []
    seq_len = 20

    for text in data:
        enc = vocab.encode(text)
        for i in range(len(enc) - 1):
            seq = enc[i:i + seq_len + 1]
            if len(seq) > 1:
                sequences.append(seq)


    split = int(len(sequences) * 0.9)
    train_seqs = sequences[:split]
    val_seqs = sequences[split:]

    train_loader = DataLoader(TextDataset(train_seqs, seq_len + 1), batch_size=64, shuffle=True)
    val_loader = DataLoader(TextDataset(val_seqs, seq_len + 1), batch_size=64)


    model = NextWordLSTM(len(vocab.word2idx)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    all_batch_losses = []
    all_batch_accs = []
    epoch_losses = []
    epoch_accs = []
    val_losses = []
    val_accs = []

    EPOCHS = 10
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        batch_losses, batch_accs, ep_loss, ep_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        all_batch_losses += batch_losses
        all_batch_accs += batch_accs
        epoch_losses.append(ep_loss)
        epoch_accs.append(ep_acc)

        v_loss, v_acc = validate(model, val_loader, criterion, device)
        val_losses.append(v_loss)
        val_accs.append(v_acc)

        print(f"Train Loss={ep_loss:.4f}, Acc={ep_acc:.4f}")
        print(f"Valid Loss={v_loss:.4f}, Acc={v_acc:.4f}")

    torch.save(model.state_dict(), model_path)
    pickle.dump(vocab, open(vocab_path, "wb"))

    print("\nSaved model & vocab!")

    # Plots

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("lstm_loss.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("lstm_accuracy.png")
    plt.close()

    print("\nSaved plots: lstm_loss.png, lstm_accuracy.png")

    interactive_cli(model, vocab, device)

if __name__ == "__main__":
    main()
