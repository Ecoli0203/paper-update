#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import html
import json
import os
import re
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import feedparser
import fitz
import markdown
import requests
from openai import OpenAI


ARXIV_API = "https://export.arxiv.org/api/query"
TOPIC_KEYWORDS = {
    "tmd": [
        "tmd",
        "transition metal dichalcogenide",
        "mos2",
        "ws2",
        "wse2",
        "mote2",
        "moire",
    ],
    "nonlinear_anomalous_hall": [
        "nonlinear anomalous hall",
        "second-order hall",
        "berry curvature dipole",
        "nonreciprocal transport",
    ],
    "topological_materials": [
        "topological",
        "weyl",
        "chern",
        "dirac semimetal",
        "surface state",
    ],
    "superconductivity": [
        "superconduct",
        "josephson",
        "pairing",
        "critical temperature",
        "twisted bilayer",
    ],
}
EVIDENCE_KEYWORDS = [
    "transport",
    "measurement",
    "xrd",
    "stm",
    "arpes",
    "phase diagram",
    "fit",
    "control experiment",
    "ab initio",
    "dft",
]
NOVELTY_KEYWORDS = ["first", "novel", "unprecedented", "new mechanism", "record", "enhanced"]


@dataclasses.dataclass
class Paper:
    arxiv_id: str
    versioned_id: str
    title: str
    summary: str
    authors: list[str]
    updated: dt.datetime
    published: dt.datetime
    abs_url: str
    pdf_url: str
    tags: list[str]
    score: float
    score_breakdown: dict[str, float]
    topic_hits: list[str]


def parse_utc(ts: str) -> dt.datetime:
    return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def env_list(key: str) -> list[str]:
    raw = os.getenv(key, "").strip()
    if not raw:
        return []
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def env_clean(key: str, default: str | None = None) -> str:
    raw = os.getenv(key)
    if raw is None:
        if default is None:
            raise KeyError(f"Missing required env: {key}")
        return default
    cleaned = raw.replace("\r", "").replace("\n", "").strip()
    if not cleaned and default is not None:
        return default
    return cleaned


def env_bool(keys: list[str], default: bool = False) -> bool:
    """Read boolean env from multiple possible names.

    This accepts a typo-compatible alias so a misnamed GitHub Secret does not
    silently force summary-only fallback mode.
    """
    for key in keys:
        raw = os.getenv(key)
        if raw is None:
            continue
        cleaned = raw.replace("\r", "").replace("\n", "").strip().lower()
        if not cleaned:
            continue
        if cleaned in {"1", "true", "yes", "on"}:
            return True
        if cleaned in {"0", "false", "no", "off"}:
            return False
    return default


def looks_like_email(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value))


def build_query() -> str:
    cats = [
        "cond-mat.mtrl-sci",
        "cond-mat.str-el",
        "cond-mat.supr-con",
        "cond-mat.mes-hall",
        "cond-mat.other",
    ]
    cat_clause = " OR ".join(f"cat:{c}" for c in cats)
    return f"({cat_clause})"


def fetch_arxiv(max_results: int = 220) -> list[dict[str, Any]]:
    params = {
        "search_query": build_query(),
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": max_results,
    }
    resp = requests.get(ARXIV_API, params=params, timeout=60)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)
    return list(feed.entries)


def count_hits(text: str, keywords: list[str]) -> int:
    return sum(1 for k in keywords if k in text)


def classify_topics(text: str) -> tuple[list[str], int]:
    hits = []
    total = 0
    for topic, kws in TOPIC_KEYWORDS.items():
        h = count_hits(text, kws)
        if h > 0:
            hits.append(topic)
            total += h
    return hits, total


def score_paper(entry: dict[str, Any], elite_authors: list[str], elite_groups: list[str]) -> Paper | None:
    title = re.sub(r"\s+", " ", entry.title).strip()
    summary = re.sub(r"\s+", " ", entry.summary).strip()
    text = normalize(f"{title} {summary}")
    authors = [a.name for a in entry.authors]
    author_text = normalize(" ".join(authors))

    topic_hits, topic_total = classify_topics(text)
    if topic_total == 0:
        return None

    elite_author_hits = count_hits(author_text, elite_authors)
    elite_group_hits = count_hits(text, elite_groups)
    lab_score = min(100.0, elite_author_hits * 28.0 + elite_group_hits * 18.0)
    topic_score = min(100.0, topic_total * 18.0)
    evidence_score = min(100.0, count_hits(text, EVIDENCE_KEYWORDS) * 16.0)
    novelty_score = min(100.0, count_hits(text, NOVELTY_KEYWORDS) * 20.0)

    overall = 0.45 * lab_score + 0.25 * topic_score + 0.20 * evidence_score + 0.10 * novelty_score
    link = entry.link
    id_part = link.split("/abs/")[-1]
    pure_id = id_part.split("v")[0]
    pdf_url = f"https://arxiv.org/pdf/{id_part}.pdf"

    return Paper(
        arxiv_id=pure_id,
        versioned_id=id_part,
        title=title,
        summary=summary,
        authors=authors,
        updated=parse_utc(entry.updated),
        published=parse_utc(entry.published),
        abs_url=link,
        pdf_url=pdf_url,
        tags=[t.term for t in getattr(entry, "tags", [])],
        score=round(overall, 2),
        score_breakdown={
            "lab": round(lab_score, 2),
            "topic": round(topic_score, 2),
            "evidence": round(evidence_score, 2),
            "novelty": round(novelty_score, 2),
        },
        topic_hits=topic_hits,
    )


def select_papers(entries: list[dict[str, Any]], lookback_hours: float, max_top: int, max_fast: int) -> tuple[list[Paper], list[Paper]]:
    elite_authors = env_list("ELITE_AUTHORS")
    elite_groups = env_list("ELITE_GROUP_KEYWORDS")
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=lookback_hours)

    papers: list[Paper] = []
    for e in entries:
        updated = parse_utc(e.updated)
        if updated < cutoff:
            continue
        paper = score_paper(e, elite_authors, elite_groups)
        if paper:
            papers.append(paper)

    papers.sort(key=lambda p: (p.score, p.updated), reverse=True)
    top = papers[:max_top]
    fast = papers[max_top : max_top + max_fast]
    return top, fast


def summarize_with_openai(api_key: str, api_base: str, model: str, paper: Paper, pdf_max_pages: int = 8) -> dict[str, str]:
    context = extract_paper_context(paper, max_pages=pdf_max_pages)
    context_text = context["text_excerpt"]
    equations = context["equations"]
    captions = context["captions"]

    sys_prompt = (
        "You are a condensed matter research assistant. "
        "Return strict JSON with keys: abstract_cn, core_idea, methods, key_formula_method, solved_problem, experiment_analysis, author_reasoning, evidence_points, figure_interpretation, confidence. "
        "Use concise Chinese. confidence must be one of: 高, 中, 低. "
        "Do not use LaTeX delimiters; convert formulas to plain text. "
        "Do not write generic templates. Every field must include paper-specific nouns from the text."
    )
    user_prompt = (
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors)}\n"
        f"Abstract: {paper.summary}\n"
        f"Topics: {', '.join(paper.topic_hits)}\n"
        "Parsed body excerpt (from PDF):\n"
        f"{context_text}\n\n"
        "Candidate equations/method expressions:\n"
        f"{equations}\n\n"
        "Figure/Table captions detected:\n"
        f"{captions}\n\n"
        "Output constraints:\n"
        "1) core_idea: 2-3句，明确本文新想法。\n"
        "2) methods: 3-6条，写清材料体系、实验/计算方法、关键参数。\n"
        "3) key_formula_method: 给出1-2个重要公式或方法表达（纯文本，不要$）。\n"
        "4) solved_problem: 必须是已解决问题，不要写建议或待确认。\n"
        "5) experiment_analysis: 至少写3个具体观测量/图中现象。\n"
        "6) author_reasoning: 必须是该文专属链路，至少包含2个具体名词（材料/器件/现象）。\n"
        "7) evidence_points: 2-4条原文证据短句（可转述但要具体）。\n"
        "8) figure_interpretation: 按 Fig1/Fig2/Fig3 分别解释物理现象与结论支撑。\n"
        "Generate detailed structured analysis for daily digest."
    )
    content = call_responses_api(
        api_key=api_key,
        api_base=api_base,
        model=model,
        input_messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    data = extract_json_object(content)
    keys = [
        "abstract_cn",
        "core_idea",
        "methods",
        "key_formula_method",
        "solved_problem",
        "experiment_analysis",
        "author_reasoning",
        "evidence_points",
        "figure_interpretation",
        "confidence",
    ]
    for k in keys:
        data.setdefault(k, "")
        if isinstance(data[k], str):
            data[k] = strip_latex_delimiters(data[k])

    if is_generic_analysis(data):
        repair_prompt = (
            "你的上一个输出过于泛化。请重写为论文特异版本。\n"
            f"Title: {paper.title}\n"
            f"Abstract: {paper.summary}\n"
            f"Body excerpt: {context_text}\n"
            "要求：所有字段都要包含本文具体名词，不得出现“建议重点看”“目标是”等空泛句。"
        )
        repaired_text = call_responses_api(
            api_key=api_key,
            api_base=api_base,
            model=model,
            input_messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": repair_prompt},
            ],
            temperature=0.1,
        )
        repaired_data = extract_json_object(repaired_text)
        for k in keys:
            if isinstance(repaired_data.get(k), str) and repaired_data[k].strip():
                data[k] = strip_latex_delimiters(repaired_data[k])
    return data


def responses_endpoint(api_base: str) -> str:
    base = api_base.strip().rstrip("/")
    if base.endswith("/responses"):
        return base
    return f"{base}/responses"


def call_responses_api(
    api_key: str,
    api_base: str,
    model: str,
    input_messages: list[dict[str, str]],
    temperature: float,
) -> str:
    endpoint = responses_endpoint(api_base)
    text_input = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in input_messages)
    payloads = []
    supports_temperature = not model.lower().startswith("gpt-5")

    base_payload = {
        "model": model,
        "input": input_messages,
    }
    if temperature is not None and supports_temperature:
        base_payload["temperature"] = temperature

    structured_payload = dict(base_payload)
    structured_payload["text"] = {"format": {"type": "json_object"}}
    payloads.append(("structured_messages_json", structured_payload))
    payloads.append(("messages", base_payload))

    string_payload = {
        "model": model,
        "input": text_input,
    }
    if temperature is not None and supports_temperature:
        string_payload["temperature"] = temperature
    string_json_payload = dict(string_payload)
    string_json_payload["text"] = {"format": {"type": "json_object"}}
    payloads.append(("string_json", string_json_payload))
    payloads.append(("string", string_payload))

    last_resp: requests.Response | None = None
    for label, payload in payloads:
        if not supports_temperature:
            payload.pop("temperature", None)
        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
        last_resp = resp
        if resp.ok:
            print(f"LLM request succeeded with payload mode: {label}")
            data = resp.json()
            return extract_response_output_text(data)
        if resp.status_code == 400 and "temperature" in resp.text and "Unsupported parameter" in resp.text:
            payload_without_temperature = dict(payload)
            payload_without_temperature.pop("temperature", None)
            resp = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload_without_temperature,
                timeout=180,
            )
            last_resp = resp
            if resp.ok:
                print(f"LLM request succeeded with payload mode: {label}_no_temperature")
                data = resp.json()
                return extract_response_output_text(data)
        if resp.status_code not in {400, 422}:
            break

    if last_resp is None:
        raise RuntimeError("LLM request was not sent")
    if last_resp.status_code == 404:
        raise RuntimeError(
            f"LLM endpoint not found: {endpoint}. Set OPENAI_API_BASE to https://opencode.ai/zen/v1 "
            "or https://opencode.ai/zen/v1/responses."
        )
    body = sanitize_error_body(last_resp.text)
    raise RuntimeError(f"LLM request failed: HTTP {last_resp.status_code} at {endpoint}. Response: {body}")


def sanitize_error_body(text: str, limit: int = 1200) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:limit]


def extract_response_output_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for item in data.get("output", []) or []:
        for part in item.get("content", []) or []:
            if part.get("type") == "output_text" and part.get("text"):
                chunks.append(str(part["text"]))
    return "\n".join(chunks)


def is_generic_analysis(data: dict[str, Any]) -> bool:
    probe = " ".join(
        [
            str(data.get("solved_problem", "")),
            str(data.get("experiment_analysis", "")),
            str(data.get("author_reasoning", "")),
        ]
    )
    generic_patterns = [
        "研究动机",
        "提出机制假设",
        "建议重点看",
        "目标是",
        "设计测量或计算",
    ]
    hit = sum(1 for g in generic_patterns if g in probe)
    return hit >= 2


def extract_paper_context(paper: Paper, max_pages: int = 8) -> dict[str, str]:
    try:
        pdf = requests.get(paper.pdf_url, timeout=90)
        pdf.raise_for_status()
    except Exception:
        return {"text_excerpt": paper.summary, "equations": "", "captions": ""}

    try:
        doc = fitz.open(stream=pdf.content, filetype="pdf")
    except Exception:
        return {"text_excerpt": paper.summary, "equations": "", "captions": ""}

    text_chunks: list[str] = []
    eq_candidates: list[str] = []
    cap_candidates: list[str] = []

    for page_idx in range(min(max_pages, len(doc))):
        page = doc[page_idx]
        page_text = page.get_text("text")
        if not page_text:
            continue

        lines = [re.sub(r"\s+", " ", ln).strip() for ln in page_text.splitlines()]
        for ln in lines:
            if len(ln) < 20:
                continue
            if re.search(r"(Fig\.?|Figure|Table)\s*\d+", ln, re.IGNORECASE):
                cap_candidates.append(ln[:220])
                continue
            if re.search(r"[=≈∝≤≥]\s*", ln) and len(ln) < 180:
                eq_candidates.append(ln)

        compact = " ".join(lines)
        compact = re.sub(r"\s+", " ", compact).strip()
        if compact:
            text_chunks.append(compact[:2200])

    excerpt = "\n\n".join(text_chunks)[:12000]
    equations = "\n".join(dedup(eq_candidates)[:8])
    captions = "\n".join(dedup(cap_candidates)[:10])
    if not excerpt:
        excerpt = paper.summary
    return {"text_excerpt": excerpt, "equations": equations, "captions": captions}


def dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        key = it.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def html_text(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def extract_output_text(resp: Any) -> str:
    output = getattr(resp, "output", None)
    if not output:
        return ""
    chunks: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if not content:
            continue
        for part in content:
            ptype = getattr(part, "type", "")
            if ptype == "output_text":
                text = getattr(part, "text", "")
                if text:
                    chunks.append(text)
    return "\n".join(chunks)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        return {}
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def summarize_rule_based(paper: Paper) -> dict[str, str]:
    abstract = re.sub(r"\s+", " ", paper.summary).strip()
    chunks = re.split(r"(?<=[.!?])\s+", abstract)
    first = chunks[0] if chunks else abstract
    second = chunks[1] if len(chunks) > 1 else ""
    topic_cn = {
        "tmd": "TMD",
        "nonlinear_anomalous_hall": "非线性反常霍尔",
        "topological_materials": "拓扑材料",
        "superconductivity": "超导",
    }
    topic_text = "、".join(topic_cn.get(t, t) for t in paper.topic_hits) if paper.topic_hits else "凝聚态"
    return {
        "abstract_cn": f"本文关注{topic_text}方向。核心内容：{first}",
        "core_idea": "未启用LLM时仅能给出摘要级信息，建议开启ENABLE_LLM=true获取逐篇细读版。",
        "methods": f"依据摘要识别的方法线索：{second or '文中通过实验表征与理论分析推进结论。'}",
        "key_formula_method": "未提取到可靠公式，请在原文方法部分查看关键表达式。",
        "solved_problem": "从摘要可见该文尝试解决特定材料体系中的机理识别或性能验证问题。",
        "experiment_analysis": "摘要模式下无法稳定还原全部实验细节。",
        "author_reasoning": "摘要模式无法完整重建作者推理链。",
        "evidence_points": "- 证据1：来自摘要主结论。\n- 证据2：来自摘要方法描述。",
        "figure_interpretation": "图表解释受限于摘要模式，建议启用LLM细读。",
        "confidence": "中",
    }


def strip_latex_delimiters(text: str) -> str:
    t = text
    t = re.sub(r"\$\$([\s\S]*?)\$\$", r"\1", t)
    t = re.sub(r"\$([^$\n]+)\$", r"\1", t)
    t = t.replace("\\(", "").replace("\\)", "")
    t = t.replace("\\[", "").replace("\\]", "")
    return t


def extract_figures(paper: Paper, output_dir: Path, limit: int = 3) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        pdf = requests.get(paper.pdf_url, timeout=90)
        pdf.raise_for_status()
    except Exception:
        return []

    images: list[tuple[int, bytes, str]] = []
    try:
        doc = fitz.open(stream=pdf.content, filetype="pdf")
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                base = doc.extract_image(xref)
                w = int(base.get("width", 0))
                h = int(base.get("height", 0))
                if w < 320 or h < 240:
                    continue
                area = w * h
                ext = base.get("ext", "png")
                data = base.get("image", b"")
                if not data:
                    continue
                images.append((area, data, ext))
    except Exception:
        return []

    if not images:
        return []
    images.sort(key=lambda x: x[0], reverse=True)

    saved: list[Path] = []
    for idx, (_, data, ext) in enumerate(images[:limit], start=1):
        path = output_dir / f"{paper.arxiv_id.replace('/', '_')}_fig{idx}.{ext}"
        path.write_bytes(data)
        saved.append(path)
    return saved


def build_markdown(
    top: list[Paper],
    fast: list[Paper],
    analyses: dict[str, dict[str, str]],
    figures: dict[str, list[Path]],
    lookback_hours: float,
    run_ts_bj: dt.datetime,
    warnings: list[str] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# 凝聚态论文日报 | {run_ts_bj.strftime('%Y-%m-%d %H:%M')} (北京时间)")
    lines.append("")
    lines.append("## 今日概览")
    lines.append(f"- 抓取窗口: 最近 {lookback_hours:.1f} 小时")
    lines.append(f"- 重点精读: {len(top)} 篇")
    lines.append(f"- 快速速览: {len(fast)} 篇")
    if warnings:
        lines.append(f"- 运行警告: {len(warnings)} 条，见文末")
    lines.append("")
    lines.append("## 今日重点精读")
    for i, paper in enumerate(top, start=1):
        a = analyses.get(paper.arxiv_id, {})
        lines.append("")
        lines.append(f"### {i}) {paper.title}")
        lines.append(f"- arXiv: {paper.abs_url}")
        lines.append(f"- 方向: {', '.join(paper.topic_hits)}")
        lines.append(f"- 摘要浓缩: {a.get('abstract_cn', '').strip()}")
        lines.append(f"- 核心想法: {a.get('core_idea', '').strip()}")
        lines.append(f"- 技术方法: {a.get('methods', '').strip()}")
        lines.append(f"- 关键公式/方法表达: {a.get('key_formula_method', '').strip()}")
        lines.append(f"- 解决问题: {a.get('solved_problem', '').strip()}")
        lines.append(f"- 实验结果分析: {a.get('experiment_analysis', '').strip()}")
        lines.append(f"- 证据要点: {a.get('evidence_points', '').strip()}")
        lines.append(f"- 图表物理现象解读: {a.get('figure_interpretation', '').strip()}")
        lines.append(f"- 作者思考路径: {a.get('author_reasoning', '').strip()}")
        lines.append(f"- 可信度: {a.get('confidence', '中')}")
        lines.append(
            "- 重要性评分: "
            f"{paper.score} (课题组 {paper.score_breakdown['lab']}, 主题 {paper.score_breakdown['topic']}, "
            f"证据 {paper.score_breakdown['evidence']}, 新颖 {paper.score_breakdown['novelty']})"
        )
        if figures.get(paper.arxiv_id):
            lines.append("- 关键图表:")
            for p in figures[paper.arxiv_id]:
                lines.append(f"  - {p.as_posix()}")

    lines.append("")
    lines.append("## 快速速览")
    lines.append("")
    lines.append("| 标题 | 方向 | 链接 |")
    lines.append("|---|---|---|")
    for paper in fast:
        lines.append(f"| {paper.title} | {', '.join(paper.topic_hits)} | {paper.abs_url} |")
    lines.append("")

    if warnings:
        lines.append("## 运行警告")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    return "\n".join(lines)


def build_html(
    top: list[Paper],
    fast: list[Paper],
    lookback_hours: float,
    run_ts_bj: dt.datetime,
    analyses: dict[str, dict[str, str]],
    figures: dict[str, list[Path]],
    warnings: list[str] | None = None,
) -> tuple[str, list[tuple[str, bytes, str]]]:
    cid_parts: list[tuple[str, bytes, str]] = []
    parts = []
    parts.append(f"<h1>凝聚态论文日报 | {html.escape(run_ts_bj.strftime('%Y-%m-%d %H:%M'))} (北京时间)</h1>")
    parts.append("<h2>今日概览</h2>")
    parts.append("<ul>")
    parts.append(f"<li>抓取窗口: 最近 {lookback_hours:.1f} 小时</li>")
    parts.append(f"<li>重点精读: {len(top)} 篇</li>")
    parts.append(f"<li>快速速览: {len(fast)} 篇</li>")
    if warnings:
        parts.append(f"<li>运行警告: {len(warnings)} 条，见文末</li>")
    parts.append("</ul>")
    parts.append("<h2>今日重点精读</h2>")

    for idx, paper in enumerate(top, start=1):
        a = analyses.get(paper.arxiv_id, {})
        parts.append(f"<h3>{idx}) {html.escape(paper.title)}</h3>")
        parts.append(f"<p><b>arXiv:</b> <a href='{html.escape(paper.abs_url)}'>{html.escape(paper.abs_url)}</a></p>")
        parts.append(f"<p><b>方向:</b> {html.escape(', '.join(paper.topic_hits))}</p>")
        parts.append(f"<p><b>摘要浓缩:</b> {html_text(a.get('abstract_cn', ''))}</p>")
        parts.append(f"<p><b>核心想法:</b> {html_text(a.get('core_idea', ''))}</p>")
        parts.append(f"<p><b>技术方法:</b> {html_text(a.get('methods', ''))}</p>")
        parts.append(f"<p><b>关键公式/方法表达:</b> {html_text(a.get('key_formula_method', ''))}</p>")
        parts.append(f"<p><b>解决问题:</b> {html_text(a.get('solved_problem', ''))}</p>")
        parts.append(f"<p><b>实验结果分析:</b> {html_text(a.get('experiment_analysis', ''))}</p>")
        parts.append(f"<p><b>证据要点:</b> {html_text(a.get('evidence_points', ''))}</p>")
        parts.append(f"<p><b>图表物理现象解读:</b> {html_text(a.get('figure_interpretation', ''))}</p>")
        parts.append(f"<p><b>作者思考路径:</b> {html_text(a.get('author_reasoning', ''))}</p>")
        parts.append(f"<p><b>可信度:</b> {html.escape(a.get('confidence', '中'))}</p>")

        imgs = figures.get(paper.arxiv_id, [])
        if not imgs:
            continue
        parts.append("<p><b>关键图表:</b></p>")
        for idx, img_path in enumerate(imgs, start=1):
            cid = f"{paper.arxiv_id.replace('/', '_')}_{idx}@daily"
            ext = img_path.suffix.lstrip(".").lower() or "png"
            ctype = "image/png" if ext == "png" else "image/jpeg"
            cid_parts.append((cid, img_path.read_bytes(), ctype))
            parts.append(f"<p>Fig.{idx}</p><img src='cid:{cid}' style='max-width:880px;width:100%;height:auto;'/>" )

    parts.append("<h2>快速速览</h2>")
    parts.append("<table border='1' cellpadding='6' cellspacing='0' style='border-collapse: collapse;'>")
    parts.append("<tr><th>标题</th><th>方向</th><th>链接</th></tr>")
    for paper in fast:
        parts.append(
            "<tr>"
            f"<td>{html.escape(paper.title)}</td>"
            f"<td>{html.escape(', '.join(paper.topic_hits))}</td>"
            f"<td><a href='{html.escape(paper.abs_url)}'>arXiv</a></td>"
            "</tr>"
        )
    parts.append("</table>")

    if warnings:
        parts.append("<h2>运行警告</h2>")
        parts.append("<ul>")
        for w in warnings:
            parts.append(f"<li>{html_text(w)}</li>")
        parts.append("</ul>")

    return "\n".join(parts), cid_parts


def llm_error_analysis(paper: Paper, err: Exception) -> dict[str, str]:
    fallback = summarize_rule_based(paper)
    msg = sanitize_error_body(str(err), limit=600)
    fallback.update(
        {
            "core_idea": "本篇 LLM 深读失败，以下为摘要级兜底结果；请查看文末运行警告定位接口问题。",
            "key_formula_method": "LLM 深读失败，未能可靠抽取公式/方法表达。",
            "experiment_analysis": f"LLM 深读失败，无法给出逐图实验分析。错误摘要：{msg}",
            "evidence_points": "LLM 深读失败，未能抽取可靠证据链。",
            "figure_interpretation": "LLM 深读失败，图表仅随文附上，未做可靠物理解读。",
            "confidence": "低",
        }
    )
    return fallback


def send_email(subject: str, html_body: str, cid_parts: list[tuple[str, bytes, str]]) -> None:
    smtp_host = env_clean("SMTP_HOST")
    smtp_port = int(env_clean("SMTP_PORT", "465"))
    smtp_user = env_clean("SMTP_USER")
    smtp_pass = env_clean("SMTP_PASS")
    from_email = env_clean("FROM_EMAIL", smtp_user)
    to_email = env_clean("TO_EMAIL")

    # QQ SMTP commonly requires sender to be the authenticated mailbox.
    if "qq.com" in smtp_host.lower():
        from_email = smtp_user

    if not looks_like_email(from_email):
        from_email = smtp_user

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content("请使用支持 HTML 的邮件客户端查看完整日报。")
    msg.add_alternative(html_body, subtype="html")

    html_part = msg.get_payload()[1]
    for cid, data, ctype in cid_parts:
        maintype, subtype = ctype.split("/", 1)
        html_part.add_related(data, maintype=maintype, subtype=subtype, cid=f"<{cid}>", filename=f"{cid}.{subtype}")

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
        server.login(smtp_user, smtp_pass)
        try:
            server.send_message(msg)
        except smtplib.SMTPSenderRefused:
            # Retry once with strict sender binding.
            msg.replace_header("From", smtp_user)
            server.send_message(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and send daily arXiv condensed matter digest")
    parser.add_argument("--lookback-hours", type=float, default=30.0)
    parser.add_argument("--max-top", type=int, default=4)
    parser.add_argument("--max-fast", type=int, default=8)
    parser.add_argument("--supplement", action="store_true")
    parser.add_argument("--pdf-max-pages", type=int, default=int(env_clean("PDF_MAX_PAGES", "8")))
    args = parser.parse_args()

    enable_llm = env_bool(["ENABLE_LLM", "EABLE_LLM"], default=False)
    model = env_clean("OPENAI_MODEL", "gpt-4o-mini")
    api_key = ""
    api_base = ""
    if enable_llm:
        api_key = env_clean("OPENAI_API_KEY")
        api_base = env_clean("OPENAI_API_BASE", "https://api.openai.com/v1")
        print(f"LLM mode: enabled; model={model}; endpoint={responses_endpoint(api_base)}")
    else:
        print("LLM mode: disabled; using rule-based fallback. Set ENABLE_LLM=true to enable deep analysis.")

    entries = fetch_arxiv()
    top, fast = select_papers(entries, args.lookback_hours, args.max_top, args.max_fast)
    if not top and not fast:
        print("No papers matched for this window.")
        return

    now_bj = dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8)))
    run_dir = Path("reports") / now_bj.strftime("%Y-%m-%d")
    image_dir = run_dir / ("supplement_images" if args.supplement else "images")
    run_dir.mkdir(parents=True, exist_ok=True)

    analyses: dict[str, dict[str, str]] = {}
    figures: dict[str, list[Path]] = {}
    warnings: list[str] = []
    for p in top:
        if enable_llm:
            try:
                analyses[p.arxiv_id] = summarize_with_openai(api_key, api_base, model, p, pdf_max_pages=args.pdf_max_pages)
            except Exception as err:
                warning = f"{p.versioned_id} LLM 深读失败：{sanitize_error_body(str(err), limit=400)}"
                print(f"WARNING: {warning}")
                warnings.append(warning)
                analyses[p.arxiv_id] = llm_error_analysis(p, err)
        else:
            analyses[p.arxiv_id] = summarize_rule_based(p)
        try:
            figures[p.arxiv_id] = extract_figures(p, image_dir, limit=3)
        except Exception as err:
            warning = f"{p.versioned_id} 图表抽取失败：{sanitize_error_body(str(err), limit=300)}"
            print(f"WARNING: {warning}")
            warnings.append(warning)
            figures[p.arxiv_id] = []

    md = build_markdown(top, fast, analyses, figures, args.lookback_hours, now_bj, warnings=warnings)
    md_name = "digest_supplement.md" if args.supplement else "digest.md"
    md_path = run_dir / md_name
    md_path.write_text(md, encoding="utf-8")

    html_body, cids = build_html(top, fast, args.lookback_hours, now_bj, analyses, figures, warnings=warnings)
    suffix = "补充更新" if args.supplement else "早报"
    subject = f"凝聚态论文日报 {now_bj.strftime('%Y-%m-%d')} | {suffix}"
    send_email(subject, html_body, cids)
    print(f"Digest sent. Markdown saved at {md_path.as_posix()}")


if __name__ == "__main__":
    main()
