# AI Next Word Prediction

## Prerequisites

- Python 3.8+
- Node.js 16+

## Installation

### Backend
```bash
pip install fastapi uvicorn torch pydantic
```

### Frontend
```bash
cd frontend
npm install
```

## Running

### Start Backend
From project root:
```bash
uvicorn api.main:app --reload
```
API runs on http://localhost:8000

### Start Frontend
In a new terminal:
```bash
cd frontend
npm run dev
```
Frontend runs on http://localhost:5173

## Usage

1. Open http://localhost:5173
2. Type partial words for autocomplete
3. Type complete words + space for AI predictions
4. Press Tab to accept suggestions

