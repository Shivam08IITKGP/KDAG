# HIVE: Hybrid Inference and Verification Engine

<div align="center">

![HIVE Architecture](image.png)

**A State-of-the-Art Consistency Verification System for Character Backstory Validation**

</div>

---

## 🚀 Overview

HIVE is a high-fidelity narrative consistency verification system that validates character backstories against canonical source texts. It employs a **multi-layered verification architecture** combining symbolic knowledge graphs, neural language models, and meta-learning arbitration.

### 🧠 The Core Logic
HIVE doesn't just "chat"—it **interrogates** claims. By combining LLM narrative intuition with strict Knowledge Graph logic and NLI (Natural Language Inference) statistical scoring, it provides authoritative, evidence-backed verdicts.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- **Python 3.10+** (Recommended: [uv](https://docs.astral.sh/uv/) for fast package management)
- **Docker Desktop** (Required for the Qdrant Vector database)
- **API Keys**: Gemini API Key and OpenRouter API Key

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
QDRANT_URL=http://localhost:6333
```

### 3. Start the Vector Database (Docker)
Open a **new terminal window** and run the following command to start Qdrant. **Keep this terminal running in the background.**
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 4. Install Dependencies
In your **original terminal**, install the project requirements:
```bash
uv pip install -r requirements.txt
```

---

## 🏃‍♂️ Running the System

HIVE operates in three distinct phases: Feature Generation, Model Training, and Batch Verification.

### Phase 1: Feature Generation (Orchestration)
Run the orchestrator to process training data, build Knowledge Graphs, and extract NLI features.
```bash
uv run main.py
```
- **Interactive Mode**: It will ask you for a **Start** and **End** index (0-based) from `utils/train.csv`.
- **Automatic Indexing**: If the book isn't indexed yet, it will automatically handle chunking and MiniLM embeddings on the first run.
- **Output**: Results are saved incrementally to `output/features_output.csv`.

### Phase 2: Train the Meta-Classifier (MANDATORY)
> **IMPORTANT**: No pretrained models are provided in this repository. To ensure full transparency and prove the system's ability to learn strictly from canonical data, you **must** train the meta-classifier locally using your generated features.
```bash
uv run ML_answering_final/train.py --all
```
- **What it does**: Trains the XGBoost and Logistic Regression arbiters strictly on the provided `utils/train.csv` features.
- **Output**: Generates local `.pkl` artifacts in `ML_answering_final/` required for Phase 3.

### Phase 3: Batch Test Pipeline
Once the models are trained, execute the end-to-end verification for your test set:
```bash
uv run test_pipeline.py --input utils/test.csv
```
- **Constraint**: This script will not function without the locally trained artifacts from Phase 2.
- **Output**: Generates `output/final_output.csv` with final verdicts (CONSISTENT vs CONTRADICTING) and detailed AI reasoning.

---

## 📊 Feature breakdown (Technical)

When running `main.py`, you will see technical metrics for every row:
- **LLM Label**: The pure narrative judgment of the model.
- **NLI Entailment**: The statistical support score from the Knowledge Graph summary.
- **NLI Contradiction**: The friction score between the claim and the character's history.
- **Backstory Embedding**: A 384-dim structural representation of the claim.

---

## 📁 Project Structure

```
hive/
├── main.py                      # Pipeline orchestrator & Interactive UI
├── test_pipeline.py             # End-to-end batch testing script
├── shared_config.py             # Unified LLM & Model configuration
├── extraction_agent/            # Phase 1: Query generation & grounding
├── graph_creator_agent/         # Phase 2: Knowledge graph synthesis
├── answering_agent/             # Phase 3: Dual Reasoning & NLI Scoring
├── ML_answering_final/          # Phase 4: Meta-learning arbitration (XGBoost)
├── Graphrag/                    # Vector DB (Qdrant) & Retrieval infra
├── utils/                       # Input data (train/test CSVs)
└── output/                      # System results and extracted features
```

---

<div align="center">

**Built with precision. Verified with rigor.**

</div>