from __future__ import annotations

from io import BytesIO
import operator
import os
import re
import time
from importlib import import_module
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

try:
    TavilySearch = import_module("langchain_tavily").TavilySearch
except (ImportError, AttributeError):
    try:
        TavilySearch = import_module(
            "langchain_community.tools.tavily_search"
        ).TavilySearchResults
    except (ImportError, AttributeError):
        TavilySearch = None

def _safe_blog_slug(value: str) -> str:
    """Keep backend output folders consistent with the Streamlit frontend."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_") or "blog"


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(
    default=275,
    description="Target word count for this section (250–300)."
)

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str = "Technical readers and software developers"
    tone: str = "Professional, technical, analytical, and practitioner-focused"
    blog_kind: Literal[
        "explainer",
        "tutorial",
        "news_roundup",
        "comparison",
        "system_design"
    ] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # keep if Tavily provides; DO NOT rely on it
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class DiagramNode(BaseModel):
    id: str = Field(description="Unique short identifier, e.g. user_query.")
    label: str = Field(description="Exact concise text to display inside the diagram node.")


class DiagramEdge(BaseModel):
    source: str
    target: str
    label: str = ""


class ImageSpec(BaseModel):
    after_heading: str = Field(
        description="Exact existing section heading after which the image should appear."
    )
    filename: str
    alt: str
    caption: str
    prompt: str = ""
    image_type: Literal["illustration", "technical_diagram"] = "illustration"
    diagram_nodes: List[DiagramNode] = Field(default_factory=list)
    diagram_edges: List[DiagramEdge] = Field(default_factory=list)
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list)


class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    image_errors: List[str]

    final: str


# -----------------------------
# 2) LLM
# -----------------------------

router_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=500
)

planner_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=1200
)

worker_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=1400
)

image_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    # Keep the requested completion below the remaining OTPM quota.
    max_tokens=500,
    max_retries=2
)


# -----------------------------
# 3) Router (decide upfront)
# -----------------------------
ROUTER_SYSTEM = """
You are an expert research routing assistant.

Decide whether a blog topic needs external research.
Return one JSON object with exactly these fields:
- needs_research: boolean
- mode: "closed_book", "hybrid", or "open_book"
- queries: list of search query strings

Use:
- closed_book for established concepts that do not depend on current facts.
- hybrid when external context would improve the article.
- open_book when the topic asks about current, recent, latest, annual,
  2026/2025 developments, current products, benchmarks, releases, or trends.

Return JSON only, without Markdown fences.
"""

def _fallback_router_decision(topic: str) -> RouterDecision:
    """
    Deterministic fallback used when the router model returns invalid JSON,
    an empty response, or a provider-side structured-output error.
    """
    topic_lower = topic.casefold()

    current_markers = (
        "2026", "2025", "2024", "latest", "current", "currently",
        "today", "this year", "recent", "state of", "trends",
        "benchmark", "release", "new model", "market"
    )

    if any(marker in topic_lower for marker in current_markers):
        return RouterDecision(
            needs_research=True,
            mode="open_book",
            queries=[
                f"{topic} latest developments",
                f"{topic} official announcements and documentation",
                f"{topic} benchmarks and industry analysis"
            ]
        )

    return RouterDecision(
        needs_research=False,
        mode="closed_book",
        queries=[]
    )


def router_node(state: State) -> dict:
    topic = state["topic"]

    try:
        decider = router_llm.with_structured_output(
            RouterDecision,
            method="json_mode"
        )

        decision = decider.invoke([
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}")
        ])

        # Guard against empty or unexpected structured responses.
        if not isinstance(decision, RouterDecision):
            decision = RouterDecision.model_validate(decision)

        print(
            f"Router decision: mode={decision.mode}, "
            f"needs_research={decision.needs_research}"
        )

    except Exception as exc:
        # Groq can occasionally return an empty/invalid JSON generation.
        # Do not terminate the entire LangGraph run for a routing failure.
        print(
            "Router model returned an invalid response. "
            f"Using deterministic fallback. Details: {type(exc).__name__}: {exc}"
        )
        decision = _fallback_router_decision(topic)

    queries = decision.queries[:3]

    # Ensure research topics always have usable search queries.
    if decision.needs_research and not queries:
        queries = [
            f"{topic} latest developments",
            f"{topic} official documentation and announcements",
            f"{topic} benchmarks and analysis"
        ]

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": queries
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily) 
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if TavilySearch is None:
        raise ImportError(
            "Tavily search requires `langchain-tavily` (recommended) or "
            "`langchain-community`. Install one with pip."
        )

    tool = TavilySearch(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:

    queries = (state.get("queries", []) or [])[:3]

    raw_results = []

    for q in queries:
        raw_results.extend(
            _tavily_search(q, max_results=3)
        )

    dedup = {}

    for r in raw_results:

        url = r.get("url")

        if not url or url in dedup:
            continue

        dedup[url] = EvidenceItem(
            title=r.get("title", ""),
            url=url,
            snippet=(r.get("snippet") or "")[:350],
            published_at=r.get("published_at"),
            source=r.get("source")
        )

    evidence = list(dedup.values())[:6]

    print(f"Queries: {len(queries)}")
    print(f"Evidence collected: {len(evidence)}")

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """
You are an expert technical blog planner.

Your task is to create a structured plan for a technical blog.

OUTPUT REQUIREMENTS:
- Return a valid JSON object.
- Do not use Markdown code fences.
- Do not include explanations outside the JSON.
- Do not wrap the response inside a "plan" key.
- Use the exact field names defined in the Plan and Task schemas.

PLAN REQUIREMENTS:
- blog_title: string
- audience: string
- tone: string
- blog_kind: one of explainer, tutorial, news_roundup, comparison, system_design
- constraints: array of strings
- tasks: array of Task objects

TASK REQUIREMENTS:
- Create exactly 4 tasks.
- Each task must contain:
    - id
    - title
    - goal
    - bullets
    - target_words
    - tags
    - requires_research
    - requires_citations
    - requires_code
- Each task must contain exactly 3 concise bullets.
- Each task should target 250–300 words.
- The complete blog should be approximately 1,000–1,200 words, excluding code and headings.
- Create substantial sections with explanations, examples, and practical context.
- Do not make sections overly brief or treat bullets as the entire answer.
- Always include target_words.
- Keep titles and goals concise.
- Do not include unnecessary descriptions.
- Return only the required JSON fields.

IMPORTANT:
- Always include audience at the top level.
- Use blog_title, not title, for the blog title.
- Do not omit any required fields.
"""

def orchestrator_node(state: State) -> dict:

    planner = planner_llm.with_structured_output(
        Plan,
        method="json_mode"
    )

    evidence = state.get("evidence", [])

    evidence_data = [
        e.model_dump() if hasattr(e, "model_dump") else e
        for e in evidence[:6]
    ]

    messages = [
        SystemMessage(content=ORCH_SYSTEM),

        HumanMessage(content=f"""
Topic: {state['topic']}

Research evidence:
{evidence_data}

Create the complete blog plan.

Return a JSON object matching the Plan schema directly.
Do not wrap the output inside a "plan" key.
Use "blog_title" instead of "title".
""")
    ]

    plan = planner.invoke(messages)

    print(plan.model_dump_json(indent=2))

    return {"plan": plan}

# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [
    e.model_dump()
    for e in state.get("evidence", [])[:3]
],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker (write one section)
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Write a substantial, detailed section of 250–300 words (±10%).
- Do not stop after briefly mentioning each bullet; explain each bullet with context, reasoning, and a concrete example where useful.
- Include relevant technical details, implications, and transitions so the section reads like a complete blog section.
- Avoid repetition and filler; depth must come from useful explanation.
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""

def worker_node(payload: dict) -> dict:
    
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
            evidence_text = "\n".join(
                f"- {e.title} | {e.url} | {e.snippet or ''}"
                for e in evidence[:3]
            )

    section_md = worker_llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words} (write approximately 250–300 words)\n"
                    f"Length instruction: This is a full-length technical blog section. Explain every bullet in depth, using examples or equations where appropriate. Do not produce a short summary.\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]

    ordered_sections = [
        md for _, md in sorted(
            state["sections"],
            key=lambda x: x[0]
        )
    ]

    body = "\n\n".join(ordered_sections).strip()

    merged_md = f"# {plan.blog_title}\n\n{body}\n"

    safe_title = _safe_blog_slug(plan.blog_title)
    blog_output_dir = Path.cwd() / "output" / safe_title
    blog_output_dir.mkdir(parents=True, exist_ok=True)

    md_path = blog_output_dir / "blog.md"
    md_path.write_text(merged_md, encoding="utf-8")

    print(f"Markdown saved: {md_path.resolve()}")

    return {
        "merged_md": merged_md,
        "final": merged_md
    }

DECIDE_IMAGES_SYSTEM = """
You are an expert technical editor and visual planner for a technical blog.

Plan 3–4 useful visuals for a typical 1,000–1,200 word article.
Aim for FOUR visuals when the content supports them:
- Usually 2 conceptual/editorial illustrations (image_type="illustration").
- Usually 2 explanatory technical diagrams (image_type="technical_diagram").
- Include at least one illustration whenever the article has a suitable
  visual concept, even when the subject is technical.
- Do not classify every technical topic as a diagram. A server room,
  GPU hardware, engineers collaborating, or an abstract AI concept is
  an illustration; architecture, flows, components, and exact relationships
  are technical diagrams.
- If four visuals would be repetitive or unsupported, return 2–3 distinct
  visuals. Never invent facts or components just to reach a count.
- Return no more than 4 visuals.

Return a compact valid JSON object matching GlobalImagePlan. JSON only.

For EVERY image include:
- after_heading: exact existing section heading without # symbols
- filename: descriptive PNG filename
- alt: concise accessible description
- caption: concise explanation
- image_type: "illustration" or "technical_diagram"
- prompt: for illustrations, a concise but detailed visual prompt (about 20–35 words);
  for technical diagrams, may be empty
- size: 1024x1024, 1024x1536, or 1536x1024
- quality: low, medium, or high

For technical_diagram images ALSO include:
- diagram_nodes: objects with unique short "id" and concise exact "label"
- diagram_edges: objects with "source", "target", and optional "label"
- Every edge source and target must match a node id.
- Include 3–8 nodes and sensible edges. Keep labels short and accurate.
- Do not invent components not supported by the blog.

For illustration images:
- diagram_nodes and diagram_edges must be empty.
- Avoid text, labels, UI screenshots, and diagrams in the generated illustration.

Place images at different relevant section headings. Do not reuse the same visual idea.
Never reproduce or rewrite the blog Markdown.
"""


def _invoke_image_planner_with_retry(planner, messages, max_attempts: int = 3):
    """Retry Groq 429 rate limits with a short exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return planner.invoke(messages)
        except Exception as exc:
            message = str(exc).casefold()
            status_code = getattr(exc, "status_code", None)
            is_rate_limit = (
                status_code == 429
                or "rate_limit_exceeded" in message
                or "rate limit reached" in message
            )
            if not is_rate_limit or attempt == max_attempts - 1:
                raise

            delay = 5 * (2 ** attempt)
            print(
                f"Image planner hit Groq rate limit. "
                f"Retrying in {delay} seconds "
                f"({attempt + 1}/{max_attempts - 1})..."
            )
            time.sleep(delay)


def decide_images(state: State) -> dict:
    planner = image_llm.with_structured_output(
        GlobalImagePlan,
        method="json_mode"
    )

    merged_md = state["merged_md"]

    messages = [
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=f"""
Topic: {state['topic']}

Plan 3–4 distinct visuals if the article supports them.
Aim for 2 FLUX illustrations and 2 Graphviz technical diagrams.
Include at least one illustration when a suitable visual concept exists.
Use different section headings. Do not make every visual a technical diagram.
Keep illustration prompts concise so the JSON response stays compact.
Return only image specifications in JSON.

BLOG:
{merged_md}
""")
    ]

    response = _invoke_image_planner_with_retry(planner, messages)

    md = merged_md
    image_specs = []

    for i, img in enumerate(response.images[:4], start=1):
        spec = img.model_dump()
        heading = spec["after_heading"].strip()

        # Find the exact existing Markdown heading.
        lines = md.splitlines()
        inserted = False

        for index, line in enumerate(lines):
            existing_heading = re.sub(
                r"^#{1,6}\s+",
                "",
                line
            ).strip()

            if existing_heading.casefold() == heading.casefold():
                placeholder = f"[[IMAGE_{i}]]"
                lines.insert(index + 1, "")
                lines.insert(index + 2, placeholder)
                md = "\n".join(lines)

                spec["placeholder"] = placeholder
                image_specs.append(spec)
                inserted = True
                break

        if not inserted:
            print(f"Image heading not found in blog: {heading}")

    image_types = [spec.get("image_type", "illustration") for spec in image_specs]
    print(
        f"Image plan selected {len(image_specs)} visuals: "
        f"{image_types.count('illustration')} FLUX illustrations, "
        f"{image_types.count('technical_diagram')} Graphviz diagrams."
    )

    return {
        "md_with_placeholders": md,
        "image_specs": image_specs
    }



def _cloudflare_generate_image_file(
    prompt: str,
    output_path: Path
) -> None:
    """Generate a decorative illustration with Cloudflare FLUX.1 schnell."""
    import base64
    import requests
    from PIL import Image

    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

    if not account_id or not api_token:
        raise RuntimeError(
            "Cloudflare credentials missing. Check CLOUDFLARE_ACCOUNT_ID "
            "and CLOUDFLARE_API_TOKEN in .env"
        )

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/"
        "@cf/black-forest-labs/flux-1-schnell"
    )
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    payload = {"prompt": prompt, "steps": 4}

    response = requests.post(url, headers=headers, json=payload, timeout=300)
    if not response.ok:
        raise RuntimeError(
            f"Cloudflare API error {response.status_code}: {response.text[:1000]}"
        )

    result = response.json()
    if not result.get("success"):
        raise RuntimeError(f"Cloudflare generation failed: {result.get('errors')}")

    image_b64 = result.get("result", {}).get("image")
    if not image_b64:
        raise RuntimeError("Cloudflare returned no image data.")

    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
        from io import BytesIO
        with Image.open(BytesIO(image_bytes)) as image:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(output_path, format="PNG")
    except Exception as exc:
        raise RuntimeError(f"Could not decode or save generated image: {exc}") from exc

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Image file was not saved correctly.")


def _generate_graphviz_diagram(spec: dict, output_path: Path) -> None:
    """Render a precise diagram from structured nodes and edges."""
    try:
        from graphviz import Digraph
    except ImportError as exc:
        raise RuntimeError(
            "Graphviz Python package is missing. Install it with: pip install graphviz"
        ) from exc

    nodes = spec.get("diagram_nodes") or []
    edges = spec.get("diagram_edges") or []

    if not 3 <= len(nodes) <= 12:
        raise ValueError(
            f"Technical diagrams need 3–12 structured nodes; received {len(nodes)}."
        )

    node_ids = {str(node["id"]) for node in nodes}
    if len(node_ids) != len(nodes):
        raise ValueError("Diagram node IDs must be unique.")

    for edge in edges:
        if str(edge["source"]) not in node_ids or str(edge["target"]) not in node_ids:
            raise ValueError(f"Diagram edge references an unknown node: {edge}")

    diagram = Digraph(name="TechnicalDiagram", format="png", engine="dot")
    diagram.attr(
        rankdir="LR",
        bgcolor="white",
        pad="0.35",
        nodesep="0.55",
        ranksep="0.75",
        splines="ortho",
        fontname="Arial",
        dpi="180",
    )
    diagram.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#EAF4FF",
        color="#2684C7",
        fontcolor="#17324D",
        penwidth="1.5",
        fontname="Arial",
        fontsize="12",
        margin="0.20,0.14",
    )
    diagram.attr(
        "edge",
        color="#2878B8",
        fontcolor="#34495E",
        penwidth="1.5",
        arrowsize="0.8",
        fontname="Arial",
        fontsize="10",
    )

    for node in nodes:
        diagram.node(str(node["id"]), label=str(node["label"]))

    for edge in edges:
        edge_attrs = {}
        if edge.get("label"):
            edge_attrs["label"] = str(edge["label"])
        diagram.edge(str(edge["source"]), str(edge["target"]), **edge_attrs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rendered_path = Path(
            diagram.render(
                filename=output_path.stem,
                directory=str(output_path.parent),
                cleanup=True,
            )
        )
        if rendered_path != output_path:
            rendered_path.replace(output_path)
    except Exception as exc:
        raise RuntimeError(
            "Graphviz could not render the diagram. Install the Graphviz "
            "system application and add its bin folder to PATH, then restart "
            f"your terminal. Original error: {exc}"
        ) from exc

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Graphviz did not create the PNG file.")


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    blog_output_dir = Path.cwd() / "output" / _safe_blog_slug(plan.blog_title)
    images_dir = blog_output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_errors: List[str] = []

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = Path(str(spec.get("filename") or "generated_image.png")).name
        if not filename.lower().endswith(".png"):
            filename = f"{Path(filename).stem}.png"

        out_path = images_dir / filename

        try:
            image_type = spec.get("image_type", "illustration")

            if not out_path.is_file() or out_path.stat().st_size == 0:
                if image_type == "technical_diagram":
                    _generate_graphviz_diagram(spec, out_path)
                else:
                    _cloudflare_generate_image_file(spec.get("prompt", ""), out_path)

            img_md = (
                f"![{spec['alt']}](images/{filename})\n"
                f"*{spec['caption']}*"
            )
            md = md.replace(placeholder, img_md)

        except Exception as exc:
            import traceback

            error_message = (
                f"{filename} ({spec.get('image_type', 'illustration')}): "
                f"{type(exc).__name__}: {exc}"
            )
            image_errors.append(error_message)
            print("\n❌ IMAGE GENERATION FAILED")
            print("Error details:", error_message)
            traceback.print_exc()

            fallback = (
                f"> **Image generation failed:** {spec.get('caption', '')}\n"
                f"> **Reason:** {type(exc).__name__}: {exc}\n"
            )
            md = md.replace(placeholder, fallback)

    md_path = blog_output_dir / "blog.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"Markdown successfully saved: {md_path.resolve()}")
    print(f"Generated image files: {len(list(images_dir.glob('*.png')))}")
    print(f"Image generation errors: {len(image_errors)}")

    return {
        "final": md,
        "image_errors": image_errors,
    }



# Build reducer subgraph with the complete image pipeline.
# merge_content -> decide_images -> generate_and_place_images

reducer_graph = StateGraph(State)

reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)

reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)

reducer_subgraph = reducer_graph.compile()

reducer_subgraph


# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app
