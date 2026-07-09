# Publish To GitHub

This guide shows how to push the current repository to your GitHub remote.

## Current Situation

If this repository has no remote yet, you can add one and push the current
branch.

Current branch is expected to be:

```text
main
```

## Before You Push

Check status:

```bash
git status --short
```

Review carefully before staging files.

Important:

- Do **not** accidentally commit large weight files such as `.pt`
- GitHub blocks files larger than 100 MB on normal pushes
- In this workspace, `yolo26n.pt` is large and should usually stay out of git

## Recommended Steps

### 1. Stage only the files you want

Example:

```bash
git add README.md
git add apps/platform/README.md
git add docs/guides/D7-evaluation-guide.md
git add docs/guides/publish-to-github.md
git add apps/platform/pyproject.toml
git add apps/platform/src/od_platform/common/result.py
git add apps/platform/src/od_platform/cli/evaluate_model.py
git add apps/platform/src/od_platform/evaluation
git add apps/platform/tests/test_evaluation.py
```

### 2. Create a commit

Example:

```bash
git commit -m "feat: add D7 model evaluation workflow"
```

### 3. Add your GitHub remote

If no remote exists:

```bash
git remote add origin https://github.com/Makeine2/ODPlat.git
```

If `origin` already exists, update it instead:

```bash
git remote set-url origin https://github.com/Makeine2/ODPlat.git
```

### 4. Push the current branch

```bash
git branch -M main
git push -u origin main
```

## If GitHub Asks For Authentication

You may need one of these:

- GitHub Desktop
- Git Credential Manager
- a Personal Access Token instead of password
- SSH remote instead of HTTPS

SSH remote form:

```bash
git remote set-url origin git@github.com:Makeine2/ODPlat.git
git push -u origin main
```

## Useful Checks

See current remotes:

```bash
git remote -v
```

See current branch:

```bash
git branch --show-current
```

See exactly what is staged:

```bash
git diff --cached --stat
```
