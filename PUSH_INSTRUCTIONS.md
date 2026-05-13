# How to push this folder to GitHub

This is a one-time setup, then a normal `git add / commit / push` cycle for updates.

## One-time setup

```bash
cd C:\022026materials\AI\repo_to_push

git init
git branch -M main
git remote add origin https://github.com/Lucas-on-the-cloud/UAV_res.git

git add .
git commit -m "Initial commit: project proposal, Week 1-3 progress report, baseline results"

git push -u origin main
```

If GitHub asks for credentials:
- Username: `Lucas-on-the-cloud`
- Password: use a **Personal Access Token** (not your GitHub password)
  - Create one at https://github.com/settings/tokens → Generate new token (classic)
  - Scopes needed: `repo`
  - Copy the token, paste it as the password

## For subsequent updates

```bash
cd C:\022026materials\AI\repo_to_push

git add .
git commit -m "Week N: describe what's new"
git push
```

## Common weekly update pattern

After each week's experiment:

1. Update `docs/progress_report.md` with new results
2. Add result JSON to `results/`
3. Export Kaggle notebook → save into `notebooks/`
4. Commit + push:
   ```bash
   git add docs/ results/ notebooks/
   git commit -m "Week N results: <one-line summary>"
   git push
   ```

## What NOT to push

The `.gitignore` already excludes:
- `*.pt`, `*.pth` model weights (too large; store in Google Drive instead)
- `data/`, `datasets/`, `visdrone_person/` (datasets)
- `runs/`, `wandb/` (training logs)
- `__pycache__/`, virtual envs, IDE files

If you accidentally `git add` a large file, undo with:
```bash
git rm --cached <file>
```
