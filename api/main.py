from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn


app = FastAPI(title="LoRA Transformer Next Word Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model_name = "distilgpt2"
lora_path = "lora_bst_best"


print("Path of lora model: ", lora_path)


tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, lora_path)
model.to(device)
model.eval()


def predict_next_words(text: str, top_k: int = 5):
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        last_logits = logits[0, -1]

        probs = torch.softmax(last_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)

    results = []
    for idx, prob in zip(top_indices.cpu().numpy(), top_probs.cpu().numpy()):
        word = tokenizer.decode([idx]).strip()
        results.append({"word": word, "prob": float(prob)})

    return results

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/predict")
def predict(text: str = Query(..., description="Input text"), k: int = 5):
    predictions = predict_next_words(text, top_k=k)
    return {"input": text, "predictions": predictions}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
