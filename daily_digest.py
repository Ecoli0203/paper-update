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


def summarize_with_openai(client: OpenAI, model: str, paper: Paper) -> dict[str, str]:
    sys_prompt = (
        "You are a condensed matter research assistant. "
        "Return strict JSON with keys: abstract_cn, methods, solved_problem, experiment_analysis, author_reasoning, figure_interpretation, confidence. "
        "Use concise Chinese. confidence must be one of: 高, 中, 低. "
        "Do not use LaTeX, do not include $ or $$ delimiters."
    )
    user_prompt = (
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors)}\n"
        f"Abstract: {paper.summary}\n"
        f"Topics: {', '.join(paper.topic_hits)}\n"
        "Generate structured analysis for daily digest."
    )
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        text={"format": {"type": "json_object"}},
    )
    content = getattr(resp, "output_text", "") or ""
    if not content:
        content = extract_output_text(resp)
    data = extract_json_object(content)
    keys = [
        "abstract_cn",
        "methods",
        "solved_problem",
        "experiment_analysis",
        "author_reasoning",
        "figure_interpretation",
        "confidence",
    ]
    for k in keys:
        data.setdefault(k, "")
        if isinstance(data[k], str):
            data[k] = strip_latex_delimiters(data[k])
    return data


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
        "methods": f"依据摘要识别的方法线索：{second or '文中主要通过理论建模/实验表征与数据分析推进结论。'}",
        "solved_problem": "目标是识别该体系中的关键物理机制并给出可验证结果，具体贡献请结合原文方法与结果段落确认。",
        "experiment_analysis": "建议重点看输运/能带/相图等关键图，验证是否有对照实验、参数扫描与替代机制排除。",
        "author_reasoning": "研究动机 -> 提出机制假设 -> 设计测量或计算 -> 用关键观测量验证 -> 讨论边界与下一步。",
        "figure_interpretation": "图表通常用于展示相图演化、输运响应或能带特征，以支撑机制解释与参数依赖关系。",
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
) -> str:
    lines: list[str] = []
    lines.append(f"# 凝聚态论文日报 | {run_ts_bj.strftime('%Y-%m-%d %H:%M')} (北京时间)")
    lines.append("")
    lines.append("## 今日概览")
    lines.append(f"- 抓取窗口: 最近 {lookback_hours:.1f} 小时")
    lines.append(f"- 重点精读: {len(top)} 篇")
    lines.append(f"- 快速速览: {len(fast)} 篇")
    lines.append("")
    lines.append("## 今日重点精读")
    for i, paper in enumerate(top, start=1):
        a = analyses.get(paper.arxiv_id, {})
        lines.append("")
        lines.append(f"### {i}) {paper.title}")
        lines.append(f"- arXiv: {paper.abs_url}")
        lines.append(f"- 方向: {', '.join(paper.topic_hits)}")
        lines.append(f"- 摘要浓缩: {a.get('abstract_cn', '').strip()}")
        lines.append(f"- 技术方法: {a.get('methods', '').strip()}")
        lines.append(f"- 解决问题: {a.get('solved_problem', '').strip()}")
        lines.append(f"- 实验结果分析: {a.get('experiment_analysis', '').strip()}")
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
    return "\n".join(lines)


def build_html(
    top: list[Paper],
    fast: list[Paper],
    lookback_hours: float,
    run_ts_bj: dt.datetime,
    analyses: dict[str, dict[str, str]],
    figures: dict[str, list[Path]],
) -> tuple[str, list[tuple[str, bytes, str]]]:
    cid_parts: list[tuple[str, bytes, str]] = []
    parts = []
    parts.append(f"<h1>凝聚态论文日报 | {html.escape(run_ts_bj.strftime('%Y-%m-%d %H:%M'))} (北京时间)</h1>")
    parts.append("<h2>今日概览</h2>")
    parts.append("<ul>")
    parts.append(f"<li>抓取窗口: 最近 {lookback_hours:.1f} 小时</li>")
    parts.append(f"<li>重点精读: {len(top)} 篇</li>")
    parts.append(f"<li>快速速览: {len(fast)} 篇</li>")
    parts.append("</ul>")
    parts.append("<h2>今日重点精读</h2>")

    for idx, paper in enumerate(top, start=1):
        a = analyses.get(paper.arxiv_id, {})
        parts.append(f"<h3>{idx}) {html.escape(paper.title)}</h3>")
        parts.append(f"<p><b>arXiv:</b> <a href='{html.escape(paper.abs_url)}'>{html.escape(paper.abs_url)}</a></p>")
        parts.append(f"<p><b>方向:</b> {html.escape(', '.join(paper.topic_hits))}</p>")
        parts.append(f"<p><b>摘要浓缩:</b> {html.escape(a.get('abstract_cn', ''))}</p>")
        parts.append(f"<p><b>技术方法:</b> {html.escape(a.get('methods', ''))}</p>")
        parts.append(f"<p><b>解决问题:</b> {html.escape(a.get('solved_problem', ''))}</p>")
        parts.append(f"<p><b>实验结果分析:</b> {html.escape(a.get('experiment_analysis', ''))}</p>")
        parts.append(f"<p><b>图表物理现象解读:</b> {html.escape(a.get('figure_interpretation', ''))}</p>")
        parts.append(f"<p><b>作者思考路径:</b> {html.escape(a.get('author_reasoning', ''))}</p>")
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

    return "\n".join(parts), cid_parts


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
    args = parser.parse_args()

    enable_llm = env_clean("ENABLE_LLM", "false").lower() in {"1", "true", "yes", "on"}
    model = env_clean("OPENAI_MODEL", "gpt-4o-mini")
    client: OpenAI | None = None
    if enable_llm:
        api_key = env_clean("OPENAI_API_KEY")
        api_base = env_clean("OPENAI_API_BASE", "https://api.openai.com/v1")
        client = OpenAI(api_key=api_key, base_url=api_base)

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
    for p in top:
        if client is not None:
            analyses[p.arxiv_id] = summarize_with_openai(client, model, p)
        else:
            analyses[p.arxiv_id] = summarize_rule_based(p)
        figures[p.arxiv_id] = extract_figures(p, image_dir, limit=3)

    md = build_markdown(top, fast, analyses, figures, args.lookback_hours, now_bj)
    md_name = "digest_supplement.md" if args.supplement else "digest.md"
    md_path = run_dir / md_name
    md_path.write_text(md, encoding="utf-8")

    html_body, cids = build_html(top, fast, args.lookback_hours, now_bj, analyses, figures)
    suffix = "补充更新" if args.supplement else "早报"
    subject = f"凝聚态论文日报 {now_bj.strftime('%Y-%m-%d')} | {suffix}"
    send_email(subject, html_body, cids)
    print(f"Digest sent. Markdown saved at {md_path.as_posix()}")


if __name__ == "__main__":
    main()
