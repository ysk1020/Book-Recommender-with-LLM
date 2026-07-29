# CLAUDE.md — Semantic Book Recommender (freeCodeCamp LLM course)

## How I want you to work with me

I'm following a course. The point is that I understand each stage well
enough to rebuild it without the video.

- **Explain before acting.** Before any file edit or state-changing bash
  command, tell me in plain language what you intend to do, which files
  you'd touch, and why. Then stop and wait for my go-ahead.
- **A question is not an instruction.** If I ask "what is a vector store?"
  or "why chunk the descriptions?", answer with an explanation only.
  Do NOT edit files or run commands in response to a question.
- **Never write the notebook cell for me** unless I explicitly ask.
  Describe what the cell needs to do and let me write it, then review it.
- **Course-first.** If the course does something a certain way, follow
  that way even if you'd do it differently. If you disagree, say so in
  one sentence and let me decide — don't silently "improve" it.
- **Explain every bash command in one line before running it.** Anything
  that installs packages, downloads data, or writes files: ask first.
- Flag anything that costs money (OpenAI API calls) before running it,
  with a rough estimate of how many calls it makes.
- Never put an API key in a tracked file. Keys live in `.env`, and
  `.env` stays in `.gitignore`.

## Project

Build a semantic book recommender: turn book descriptions into
embeddings, retrieve by meaning rather than keywords, then filter
results by category and emotional tone, exposed through a dashboard.

## Stack

- Python, Jupyter notebooks, pandas / numpy
- LangChain (+ community, OpenAI, and Chroma integrations)
- Hugging Face `transformers` for zero-shot classification and
  sentiment analysis
- Chroma as the vector store
- Gradio for the dashboard
- matplotlib / seaborn for the exploratory plots
- Kaggle books dataset

## Pipeline stages

1. Data cleaning — missingness patterns, drop short descriptions,
   check category counts
2. Text splitting + embedding, build the Chroma vector database
3. Retrieval — semantic search over descriptions
4. Zero-shot classification — simplify categories, then check
   classifier accuracy
5. Sentiment / emotion extraction per description
6. Gradio dashboard — query box + category filter + tone filter

Don't skip ahead to a later stage without asking me.

## Conventions

- One notebook per stage, cleaned intermediate data saved to CSV so a
  later stage never re-runs an earlier one
- Any step that hits an API or a model gets cached to disk — I should
  never pay twice for the same computation