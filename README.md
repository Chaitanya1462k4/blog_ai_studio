# ✍️ AI Blog Studio

> **Enter a topic. A multi-stage LLM pipeline writes, illustrates, and delivers a complete technical blog.**

🚀 **Live Demo:** [Try AI Blog Studio](https://blogaistudio.streamlit.app/)

AI Blog Studio is a graph-orchestrated content pipeline built with **LangGraph** and powered by **Groq LLMs**. Given any topic, it intelligently routes through research, planning, parallel section writing, and automatic image generation — producing a polished, illustrated blog post with AI-generated artwork and technical diagrams.

---

## 🧠 How It Works

The pipeline is a **LangGraph `StateGraph`** with conditional routing, parallel fan-out, and a reducer subgraph:

```
User Topic
    │
    ▼
┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────┐ ┌──────────┐
│  ROUTER  │────▶│ RESEARCH │────▶│ ORCHESTRATOR │────▶│ WORKER 1 │ │ WORKER 2 │
│          │     │(optional)│     │  (Planner)   │     │          │ │          │
│ Classifies     │          │     │              │  ┌──│ Writes   │ │ Writes   │
│ topic &  │     │ Searches │     │ Creates a    │  │  │ Section 1│ │ Section 2│
│ decides  │     │ Tavily   │     │ structured   │  │  └──────────┘ └──────────┘
│ research │     │ for      │     │ blog plan    │──┤       Fan-out (parallel)
│ need     │     │ sources  │     │ with tasks   │  │  ┌──────────┐ ┌──────────┐
└──────────┘     └──────────┘     └──────────────┘  │  │ WORKER 3 │ │ WORKER 4 │
      │                                             └──│          │ │          │
      │ (closed_book)                                  │ Writes   │ │ Writes   │
      └──── skips research ──────▶ ORCHESTRATOR        │ Section 3│ │ Section 4│
                                                       └──────┬───┘ └────┬─────┘
                                                              │          │
                                                              ▼          ▼
                                                    ┌─────────────────────────┐
                                                    │    REDUCER SUBGRAPH     │
                                                    │                         │
                                                    │  Merge → Image Plan →   │
                                                    │  Generate Images →      │
                                                    │  Final Blog + PNGs      │
                                                    └─────────────────────────┘
```

| Step | Type | Component | What it does |
|------|------|-----------|-------------|
| 1 | 🏷️ **Classifier** | Router | Decides `closed_book`, `hybrid`, or `open_book` — skips research for well-known topics |
| 2 | 🔍 **Function** | Research | Searches Tavily (up to 3 queries × 3 results), deduplicates by URL |
| 3 | 📋 **Planner** | Orchestrator | Creates a structured `Plan` with title, audience, tone, and 4 section tasks |
| 4 | ✏️ **LLM Calls** | Workers (×4) | Write sections **in parallel** via LangGraph `Send()` fan-out |
| 5 | 🖼️ **Subgraph** | Reducer | Merges sections → plans 3–4 visuals → generates images → saves final blog |

### What generates the images?

| Visual Type | Generator | How |
|---|---|---|
| **Illustrations** | Cloudflare FLUX-1-schnell | AI-generated artwork from LLM-crafted prompts |
| **Technical Diagrams** | Graphviz | Structured nodes & edges planned by LLM, rendered as precise diagrams |

---

## ⚡ Key Features

- **Conditional routing** — Router skips web research for well-known topics, saving time and API calls
- **Parallel writing** — 4 workers write blog sections concurrently via LangGraph `Send()`
- **Automatic image generation** — LLM plans visuals, then generates AI illustrations (FLUX) and technical diagrams (Graphviz)
- **Structured planning** — Pydantic-validated `Plan` with title, audience, tone, blog type, and per-section goals
- **Blog management** — Save, load, and browse previously generated blogs
- **Export options** — Download as Markdown or as a ZIP bundle with all images
- **Inline image rendering** — Blog preview displays generated images alongside text
- **Fallback & retry logic** — Graceful handling of rate limits and generation failures

---

## 🏗️ Architecture

> **Note:** This is a **multi-stage LLM workflow**, not a multi-agent system. None of the components autonomously decide to use tools in a reasoning loop — the LangGraph `StateGraph` controls all routing, parallelism, and data flow.

```
┌────────────────────────────────────────────────────────┐
│                 Streamlit UI (app.py)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Sidebar  │  │ Progress │  │ Tabbed   │  │Download│ │
│  │ (topic,  │  │ bar &    │  │ results  │  │ .md /  │ │
│  │  saved   │  │ status)  │  │ (5 tabs) │  │ .zip   │ │
│  │  blogs)  │  │          │  │          │  │        │ │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘ │
└───────────────────────┬────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────┐
│              LangGraph StateGraph (backend.py)          │
│                                                         │
│  ┌────────┐  ┌────────┐  ┌────────────┐  ┌──────────┐ │
│  │ Router │─▶│Research│─▶│Orchestrator│─▶│Workers×4 │ │
│  │        │  │        │  │  (Planner) │  │(parallel)│ │
│  └────────┘  └────────┘  └────────────┘  └────┬─────┘ │
│                                                │       │
│  ┌─────────────────────────────────────────────▼─────┐ │
│  │           Reducer Subgraph                        │ │
│  │  merge_content → decide_images → generate_images  │ │
│  └───────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
         │                    │                  │
    ┌────▼────┐         ┌────▼─────┐       ┌────▼────┐
    │  Groq   │         │  Tavily  │       │Cloudflare│
    │  LLMs   │         │  Search  │       │FLUX + GV │
    └─────────┘         └──────────┘       └──────────┘
```

---

## 📁 Project Structure

```
blog_ai_studio/
├── app.py             # Streamlit UI — topic input, progress, tabbed results
├── backend.py         # LangGraph StateGraph — all pipeline stages and schemas
├── Dockerfile         # Container-ready deployment
├── requirements.txt   # Python dependencies
├── .env               # API keys (not committed)
├── .gitignore
└── output/            # Generated blogs (each in its own folder)
    └── Blog_Title/
        ├── blog.md
        └── images/
            ├── illustration.png
            └── diagram.png
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd blog_ai_studio
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Graphviz system package is also required for technical diagrams. Install it from [graphviz.org](https://graphviz.org/download/) and add it to PATH.

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
```

| Variable | Where to get it |
|----------|----------------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) |
| `CLOUDFLARE_ACCOUNT_ID` | [dash.cloudflare.com](https://dash.cloudflare.com) → Account ID in sidebar |
| `CLOUDFLARE_API_TOKEN` | [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens) |

---

## ▶️ Running the App

### Option A — Streamlit (Recommended)

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`. Enter a topic, click **🚀 Generate blog**, and watch the pipeline run.

### Option B — Docker

```bash
docker build -t blog-ai-studio .
docker run -p 8501:8501 --env-file .env blog-ai-studio
```

---

## 🔍 Pipeline Stage Details

### 🏷️ Router (`backend.py`)
- **Type:** LLM classifier (single structured output call)
- Classifies the topic into `closed_book`, `hybrid`, or `open_book` mode.
- Uses a small fast model (`openai/gpt-oss-20b`) for cost efficiency.
- Includes a deterministic fallback if the LLM returns invalid JSON.

### 🔍 Research (`backend.py`)
- **Type:** Python function (no LLM involved)
- Runs up to 3 Tavily search queries with 3 results each.
- Deduplicates by URL and validates into `EvidenceItem` Pydantic models.
- Skipped entirely if the Router classifies the topic as `closed_book`.

### 📋 Orchestrator (`backend.py`)
- **Type:** LLM planner (single structured output call)
- Generates a `Plan` with: blog title, audience, tone, blog kind, constraints, and 4 `Task` objects.
- Each task has: title, goal, 3–6 bullets, word target (250–300), and flags for research/citations/code.
- Uses `qwen/qwen3.8-27b` with JSON mode for reliable structured output.

### ✏️ Workers (`backend.py`)
- **Type:** LLM calls (one per section, run in parallel)
- Each worker receives the full plan + its assigned task + evidence.
- Writes 250–300 words of Markdown with section heading, explanations, and code if flagged.
- Fan-out via LangGraph `Send()` — all 4 run concurrently and merge via `operator.add` reducer.

### 🖼️ Reducer Subgraph (`backend.py`)
A compiled sub-`StateGraph` with 3 stages:

1. **merge_content** — orders sections by task ID, prepends blog title, saves initial `blog.md`
2. **decide_images** — LLM plans 3–4 visuals (illustrations + diagrams), inserts placeholders into Markdown
3. **generate_and_place_images** — generates FLUX illustrations via Cloudflare API and Graphviz diagrams, replaces placeholders with image references

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (`StateGraph`, `Send`, subgraphs) |
| LLM | Groq — `gpt-oss-20b` (router), `qwen3.8-27b` (planner/image), `gpt-oss-120b` (writer) |
| Web Search | Tavily (`langchain-tavily`) |
| AI Images | Cloudflare Workers AI — FLUX-1-schnell |
| Diagrams | Graphviz (Python `graphviz` + system binary) |
| Schemas | Pydantic v2 (structured output validation) |
| UI | Streamlit |
| Deployment | Docker |
| Environment | `python-dotenv` |

---

## 📦 Key Dependencies

```
langgraph
langchain-groq
langchain-core
langchain-tavily
pydantic >= 2.5.0
streamlit
graphviz
Pillow
requests
python-dotenv
```

> See [`requirements.txt`](./requirements.txt) for the full list.

---

## 🖥️ UI Tabs

| Tab | Content |
|-----|---------|
| 📝 **Article** | Full blog preview with inline images, download as `.md` or `.zip` bundle |
| 🧩 **Plan** | Audience, tone, blog type, and section task breakdown table |
| 🔎 **Research Sources** | Titles, publishers, dates, and clickable URLs from Tavily |
| 🖼️ **Images** | Image plan JSON + generated image gallery (2-column grid) |
| 🧾 **Run Logs** | Raw LangGraph state updates for debugging |

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute it.
