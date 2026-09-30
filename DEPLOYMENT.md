# Deployment

Target: **Streamlit Community Cloud** (free). No Docker, no paid API, no secret keys,
no external database.

---

## Before you start

Confirm the project runs on your machine:

```bash
pip install -r requirements.txt
pytest
streamlit run app/streamlit_app.py
```

If the tests pass and the app opens, it will deploy.

---

## Step 1 - Create a GitHub repository

Create a new **public** repository on GitHub. Community Cloud can deploy private repos
too, but public is simpler for a college project.

Do not initialise it with a README; this project already has one.

## Step 2 - Push the project

```bash
cd resume-job-matcher

git init
git add .
git commit -m "NLP resume-job matcher"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Check that `data/` was pushed. The app needs `data/skills.csv` and `data/aliases.csv`
at runtime; without them it raises a clear `FileNotFoundError` on startup. The
`.gitignore` excludes generated reports and downloaded models, not data.

## Step 3 - Open Streamlit Community Cloud

Go to <https://share.streamlit.io> and sign in with GitHub. Authorise access to your
repositories when prompted. If you see a "Connect GitHub account" warning in your
workspace, click it and complete the GitHub authentication prompts.

## Step 4 - Create the app

In the upper-right corner of your workspace, click **Create app**.

When asked "Do you already have an app?", click **Yup, I have an app**.

## Step 5 - Fill in the three fields

| Field | Value |
|---|---|
| Repository | `<your-username>/<your-repo>` |
| Branch | `main` |
| Main file path | `app/streamlit_app.py` |

The main file path matters most. Community Cloud defaults to `streamlit_app.py` at the
repository root, which does not exist in this project — the file lives inside `app/`.

Community Cloud suggests repositories and files, but the suggestions are not always
complete. If yours is not listed, type it in manually.

Optionally set a custom subdomain under **App URL** (for example `resume-job-matcher`),
which gives you a memorable link instead of an auto-generated one.

Under **Advanced settings**, set the Python version to **3.11** or **3.12**.

## Step 6 - Deploy

Click **Deploy**. The first build takes three to five minutes while pip installs the
dependencies. Your app appears at:

```
https://<your-subdomain>.streamlit.app
```

Open it, choose "Use a sample" for both the resume and the job description, and click
**Analyse match** to confirm the deployment works end to end.

Pushing a new commit to `main` redeploys automatically.

---

## What gets installed

`requirements.txt` installs only what the app needs: Streamlit, scikit-learn, pandas,
NumPy, SciPy, PyMuPDF and python-docx. All of them ship prebuilt wheels, so nothing
compiles during the build.

`packages.txt` is present but **deliberately empty** (zero bytes), because no apt system
libraries are required. Community Cloud passes each line of this file to `apt-get`, and
comment lines are not documented as supported — a stray `# note` can be read as a package
name and fail the build. Keep the file empty, or add real package names one per line if
you later need a system library.

## Memory: read this before enabling sentence-transformers

The free Community Cloud tier gives each app roughly **1 GB of RAM**.

`sentence-transformers` pulls in PyTorch. Installed, that is around 800 MB of disk, and
loading the model plus the Streamlit runtime pushes memory close to the ceiling. Apps
that exceed it are killed and restart in a loop.

The project is therefore configured to deploy **without** it. Semantic similarity uses
the LSA backend (TF-IDF + Truncated SVD), which needs no download and no extra memory.
The application reports which backend produced each score, in the sidebar and in the
dashboard, so nothing is hidden.

If you want to try the transformer backend anyway:

1. Uncomment these lines in `requirements.txt`:
   ```
   sentence-transformers>=2.7,<4.0
   ```
2. Redeploy and watch the logs on first load. If you see the app restart repeatedly or
   a "Killed" / out-of-memory message, revert the change.

A safer place to run the transformer backend is locally, or on a host with more memory
(Hugging Face Spaces, Render, or a small VM).

## Optional: spaCy on Community Cloud

spaCy itself is light, but its model is not on PyPI. To install it, uncomment **both**
lines in `requirements.txt`:

```
spacy>=3.7,<4.0
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Do not use `python -m spacy download` in a deployment: there is no shell step in the
Community Cloud build, and the download would run on every cold start. The wheel URL in
`requirements.txt` is the supported approach.

Without spaCy the app runs fine on rule-based fallbacks.

## Configuration via environment variables

Set these under **Settings -> Secrets** (or as shell variables locally):

| Variable | Values | Effect |
|---|---|---|
| `RJM_SEMANTIC_BACKEND` | `auto`, `st`, `lsa` | Force a semantic backend. `lsa` guarantees no download. |
| `RJM_SEMANTIC_MODEL` | any model id | Change the sentence-transformer model. |

Example, to pin the fallback and make cold starts fast:

```toml
RJM_SEMANTIC_BACKEND = "lsa"
```

## Running it elsewhere

**Locally on any machine**

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

**Hugging Face Spaces** — create a Streamlit Space, push the same files, and rename
`app/streamlit_app.py` to `app.py` at the root (or add a one-line `app.py` that imports
and calls `main`). Spaces offers more memory, so the transformer backend is viable.

**Docker** (not required, included for completeness)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `FileNotFoundError: Skill database not found` | `data/` was not pushed. Check `git status` and commit `data/skills.csv` and `data/aliases.csv`. |
| App builds but shows "main file not found" | The main file path is wrong. It must be `app/streamlit_app.py`. |
| App restarts in a loop, logs show "Killed" | Out of memory, almost always from `sentence-transformers`. Comment it out of `requirements.txt` and redeploy. |
| Semantic score looks lower than expected | You are on the LSA backend. The sidebar reports which backend is active; the values are calibrated but not identical to transformer scores. |
| "Almost no text could be read from this PDF" | The PDF is a scan (an image). Upload a text-based PDF or DOCX, or paste the text. |
| Build fails compiling a wheel | You changed a dependency to a version without a prebuilt wheel. Pin it back to the range in `requirements.txt`. |
| Uploads rejected as too large | The app caps uploads at 5 MB (`MAX_FILE_SIZE_MB` in `src/config.py`). |

## Deployment checklist

- [ ] `pytest` passes locally
- [ ] `streamlit run app/streamlit_app.py` opens and analyses a sample pair
- [ ] `data/skills.csv` and `data/aliases.csv` are committed
- [ ] `requirements.txt` is committed and contains no unused packages
- [ ] `packages.txt` is committed
- [ ] Main file path is set to `app/streamlit_app.py`
- [ ] Python version set to 3.11 or 3.12
- [ ] The deployed app analyses the bundled samples successfully
- [ ] No API keys or secrets anywhere in the repository
- [ ] No real personal resumes committed
