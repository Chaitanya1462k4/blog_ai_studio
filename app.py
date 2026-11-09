from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load API keys before backend import because ChatGroq initializes immediately.
load_dotenv()

# backend.py must be in the same folder as this file.
from backend import app as graph_app


# -----------------------------
# Configuration and helpers
# -----------------------------
st.set_page_config(
    page_title="AI Blog Studio",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUT_DIR = Path("output")
IMAGES_DIR = OUTPUT_DIR / "images"


def safe_slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return value.strip("_") or "blog"


def as_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [as_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: as_dict(item) for key, item in value.items()}
    return value


def extract_title(markdown: str, fallback: str = "Untitled blog") -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def blog_folder_for_title(title: str) -> Path:
    return OUTPUT_DIR / safe_slug(title)


def list_saved_blogs() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = list(OUTPUT_DIR.glob("*/blog.md")) + list(OUTPUT_DIR.glob("*.md"))
    files = [path for path in files if path.is_file()]
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


def read_blog_title(path: Path) -> str:
    markdown = path.read_text(encoding="utf-8", errors="replace")
    fallback = path.parent.name if path.name == "blog.md" else path.stem
    return extract_title(markdown, fallback)

def make_bundle(markdown: str, filename: str, blog_dir: Path) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(filename, markdown.encode("utf-8"))
        images_dir = blog_dir / "images"
        if images_dir.exists():
            for image_path in images_dir.rglob("*"):
                if image_path.is_file():
                    archive.write(image_path, arcname=str(Path("images") / image_path.relative_to(images_dir)))
    return buffer.getvalue()


def render_markdown(markdown: str, blog_dir: Path) -> None:
    image_pattern = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
    cursor = 0
    for match in image_pattern.finditer(markdown):
        before = markdown[cursor:match.start()]
        if before.strip():
            st.markdown(before)

        alt = match.group("alt").strip()
        src = match.group("src").strip()
        if src.startswith(("http://", "https://")):
            st.image(src, caption=alt or None, use_container_width=True)
        else:
            relative = Path(src.lstrip("./"))
            candidates = [blog_dir / relative, blog_dir / "images" / relative.name]
            image_path = next((p for p in candidates if p.is_file()), None)
            if image_path:
                st.image(str(image_path), caption=alt or None, use_container_width=True)
            else:
                st.warning(f"Image not found for this blog: `{src}`")
        cursor = match.end()

    remainder = markdown[cursor:]
    if remainder.strip():
        st.markdown(remainder)

def normalize_plan(plan_value: Any) -> dict | None:
    if not plan_value:
        return None
    result = as_dict(plan_value)
    return result if isinstance(result, dict) else None


# -----------------------------
# Session state
# -----------------------------
if "last_out" not in st.session_state:
    st.session_state.last_out = None
if "event_logs" not in st.session_state:
    st.session_state.event_logs = []
if "selected_blog" not in st.session_state:
    st.session_state.selected_blog = None


# -----------------------------
# Header
# -----------------------------
st.title("✍️ AI Blog Studio")
st.caption("A multi-agent AI blog writer powered by LangGraph, Groq, Tavily, and Cloudflare FLUX image generation, with Graphviz-powered technical diagrams.")

generate = False
with st.sidebar:
    st.header("Create a blog")
    topic = st.text_area(
        "What would you like to write about?",
        placeholder="e.g. RAG architectures and their use cases",
        height=120,
    )
    as_of = st.date_input("Research as of", value=date.today())
    generate = st.button("🚀 Generate blog", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Saved blogs")
    saved_blogs = list_saved_blogs()

    if saved_blogs:
        blog_labels = {
            f"{read_blog_title(p)} · {p.parent.name if p.name == 'blog.md' else p.name}": p
            for p in saved_blogs[:50]
        }
        selected_label = st.selectbox(
            "Choose a previous blog",
            options=list(blog_labels.keys()),
            index=None,
            placeholder="Select a saved blog",
        )
        if st.button("📂 Load blog", use_container_width=True, disabled=not selected_label):
            selected_path = blog_labels[selected_label]
            markdown = selected_path.read_text(encoding="utf-8", errors="replace")
            st.session_state.last_out = {
                "topic": read_blog_title(selected_path),
                "blog_dir": str(selected_path.parent),
                "plan": None,
                "evidence": [],
                "image_specs": [],
                "image_errors": [],
                "final": markdown,
            }
            st.rerun()
    else:
        st.caption("Your generated Markdown files will appear here.")


# -----------------------------
# Run LangGraph
# -----------------------------
if generate:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        inputs = {
            "topic": topic.strip(),
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "image_errors": [],
            "final": "",
        }

        st.session_state.event_logs = []
        status = st.status("Starting your blog workflow…", expanded=True)
        progress = st.progress(0)
        progress_text = st.empty()
        latest_state = None

        # Stream once. The final values event contains the accumulated graph state.
        # Avoid invoking the graph a second time after streaming.
        try:
            for index, state_update in enumerate(
                graph_app.stream(inputs, stream_mode="values")
            ):
                latest_state = state_update
                if isinstance(state_update, dict):
                    mode = state_update.get("mode", "—")
                    evidence_count = len(state_update.get("evidence") or [])
                    section_count = len(state_update.get("sections") or [])
                    image_count = len(state_update.get("image_specs") or [])
                    plan = state_update.get("plan")
                    task_count = len(getattr(plan, "tasks", []) or []) if plan else 0

                    progress_text.markdown(
                        f"**Mode:** `{mode}`  ·  "
                        f"**Research sources:** {evidence_count}  ·  "
                        f"**Sections:** {section_count}/{task_count or 4}  ·  "
                        f"**Planned images:** {image_count}"
                    )
                progress.progress(min(95, 10 + index * 12))
                st.session_state.event_logs.append(
                    f"Graph update {index + 1}: "
                    + json.dumps(as_dict(state_update), ensure_ascii=False, default=str)[:2500]
                )

            if not isinstance(latest_state, dict):
                raise RuntimeError("The LangGraph workflow returned no state.")

            latest_plan = latest_state.get("plan")
            latest_plan_dict = as_dict(latest_plan) if latest_plan else {}
            generated_title = latest_plan_dict.get("blog_title") if isinstance(latest_plan_dict, dict) else None
            latest_state["blog_dir"] = str(blog_folder_for_title(generated_title)) if generated_title else str(OUTPUT_DIR)
            st.session_state.last_out = latest_state
            progress.progress(100)
            status.update(label="✅ Blog generation completed", state="complete", expanded=False)
            st.rerun()

        except Exception as exc:
            status.update(label="❌ Blog generation failed", state="error", expanded=True)
            st.session_state.event_logs.append(f"{type(exc).__name__}: {exc}")
            st.exception(exc)


# -----------------------------
# Results
# -----------------------------
out = st.session_state.last_out

if out:
    plan_dict = normalize_plan(out.get("plan"))
    evidence = [as_dict(item) for item in (out.get("evidence") or [])]
    image_specs = [as_dict(item) for item in (out.get("image_specs") or [])]
    final_markdown = out.get("final") or out.get("merged_md") or ""

    title = (
        (plan_dict or {}).get("blog_title")
        or extract_title(final_markdown)
    )
    blog_dir = Path(out.get("blog_dir") or blog_folder_for_title(title))
    blog_images_dir = blog_dir / "images"

    st.divider()
    st.subheader(title)
    st.caption("Generated blog")

    tab_preview, tab_plan, tab_sources, tab_images, tab_logs = st.tabs(
        ["📝 Article", "🧩 Plan", "🔎 Research sources", "🖼️ Images", "🧾 Run logs"]
    )

    with tab_preview:
        if final_markdown:
            render_markdown(final_markdown, blog_dir)
            filename = f"{safe_slug(title)}.md"
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=final_markdown.encode("utf-8"),
                    file_name=filename,
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    "📦 Download blog + images",
                    data=make_bundle(final_markdown, filename, blog_dir),
                    file_name=f"{safe_slug(title)}_bundle.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
        else:
            st.info("The graph did not return final Markdown.")

    with tab_plan:
        if plan_dict:
            st.write("**Audience:**", plan_dict.get("audience", "—"))
            st.write("**Tone:**", plan_dict.get("tone", "—"))
            st.write("**Blog type:**", plan_dict.get("blog_kind", "—"))
            tasks = plan_dict.get("tasks") or []
            if tasks:
                st.dataframe(
                    pd.DataFrame(tasks),
                    use_container_width=True,
                    hide_index=True,
                )
            if plan_dict.get("constraints"):
                with st.expander("Writing constraints"):
                    st.write(plan_dict["constraints"])
        else:
            st.info("No planning data is available for this saved article.")

    with tab_sources:
        if evidence:
            source_rows = []
            for item in evidence:
                source_rows.append({
                    "Title": item.get("title", ""),
                    "Source": item.get("source", ""),
                    "Published": item.get("published_at", ""),
                    "URL": item.get("url", ""),
                })
            st.dataframe(
                pd.DataFrame(source_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "URL": st.column_config.LinkColumn("URL", display_text="Open source")
                },
            )
        else:
            st.info("No research sources were returned for this run.")

    with tab_images:
        if image_specs:
            st.markdown("**Image plan**")
            st.json(image_specs)
        image_files = sorted(blog_images_dir.glob("*")) if blog_images_dir.exists() else []
        if image_files:
            columns = st.columns(2)
            for index, image_path in enumerate(image_files):
                with columns[index % 2]:
                    st.image(str(image_path), caption=image_path.name, use_container_width=True)
        elif not image_specs:
            st.info("No images were planned for this article.")
        else:
            st.warning("Images were planned, but no generated image files were found.")

        image_errors = out.get("image_errors") or []
        if image_errors:
            st.error("One or more images could not be generated.")
            for image_error in image_errors:
                st.code(image_error)

    with tab_logs:
        if st.session_state.event_logs:
            st.text_area(
                "Workflow events",
                value="\n\n".join(st.session_state.event_logs),
                height=450,
            )
        else:
            st.info("No run logs are available.")

else:
    st.info("Enter a topic in the sidebar and select **Generate blog** to get started.")

st.markdown("---")
st.caption("AI Blog Studio · Built with Streamlit and LangGraph")
