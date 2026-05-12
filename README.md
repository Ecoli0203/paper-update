# Condensed Matter Daily Digest (arXiv)

Automated daily digest for condensed matter papers focused on:

- TMD materials
- Nonlinear anomalous Hall
- Topological materials
- Superconductivity

It runs on GitHub Actions and sends email reports with:

- Structured analysis per top paper (summary, methods, solved problem, experiment analysis, reasoning path)
- PDF-aware deep analysis (uses parsed body text/captions/equation candidates, not abstract-only)
- arXiv links
- Key figures extracted from PDF (2-3 per top paper)

## 1) Required GitHub Secrets

Set these in `Settings -> Secrets and variables -> Actions`.

- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_API_BASE`: API base URL ending with `/v1` (for OpenCodeZen use `https://opencode.ai/zen/v1`)
- `OPENAI_MODEL`: e.g. `gpt-4o-mini`
- `ENABLE_LLM`: `true` to enable OpenAI summaries, `false` to use rule-based fallback
- `SMTP_USER`: your QQ email address
- `SMTP_PASS`: QQ SMTP authorization code (not login password)
- `FROM_EMAIL`: sender email (usually same as `SMTP_USER`)
- `TO_EMAIL`: receiver email (for you: `ecoli_2022@qq.com`)
- `ELITE_AUTHORS`: comma-separated important author names (optional but recommended)
- `ELITE_GROUP_KEYWORDS`: comma-separated group/institute keywords (optional)
- `PDF_MAX_PAGES`: how many PDF pages to parse for deep analysis (optional, default `8`)
- `MAX_TOP`: number of fully analyzed papers in digest (optional, default `4`)
- `MAX_FAST`: number of fast-scan papers (optional, default `8`)

Example `ELITE_AUTHORS`:

```text
andrea young,pablo jarillo-herrero,ashvin vishwanath,long ju,
eva andrei,jie shan,kin fai mak,yoshinori tokura
```

Example `ELITE_GROUP_KEYWORDS`:

```text
mit,harvard,stanford,tsinghua,pku,princeton,max planck
```

## 2) Schedule

Workflow is configured in `.github/workflows/daily-digest.yml`:

- `10 0 * * *` UTC (08:10 Beijing): main digest
- `10 1 * * *` UTC (09:10 Beijing): supplement update window

Manual run is available via `workflow_dispatch`.

## LLM mode notes

- If you do not have a standalone OpenAI key (for example only using Chat UI access), set `ENABLE_LLM=false`.
- In fallback mode, the pipeline still runs and sends digest email, but analysis quality is lower than full LLM mode.
- The workflow log prints `LLM mode: enabled` or `LLM mode: disabled`; check this line after manual runs.

## 3) Local test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="..."
export OPENAI_API_BASE="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
export SMTP_HOST="smtp.qq.com"
export SMTP_PORT="465"
export SMTP_USER="your@qq.com"
export SMTP_PASS="qq_smtp_auth_code"
export FROM_EMAIL="your@qq.com"
export TO_EMAIL="ecoli_2022@qq.com"
export ELITE_AUTHORS="andrea young,pablo jarillo-herrero"
export ELITE_GROUP_KEYWORDS="mit,harvard,stanford"

python daily_digest.py --lookback-hours 30
```

## 4) Output

- Markdown report: `reports/YYYY-MM-DD/digest.md`
- Supplement report: `reports/YYYY-MM-DD/digest_supplement.md`
- Extracted images: `reports/YYYY-MM-DD/images/` and `reports/YYYY-MM-DD/supplement_images/`
