# Git Workflow

This project follows a simplified Git Flow workflow.

## Branch Structure

- **main** - Production-ready code, stable releases only
- **dev** - Main development branch, integration happens here
- **feature/** - Feature branches (e.g., `feature/data-preprocessing`)
- **bugfix/** - Bug fix branches (e.g., `bugfix/fix-missing-values`)

## Workflow

### Starting a new feature

```bash
# Make sure you're on dev and it's up to date
git checkout dev
git pull origin dev

# Create a new feature branch
git checkout -b feature/your-feature-name
```

### Working on your feature

```bash
# Make changes and commit regularly
git add .
git commit -m "descriptive commit message"

# Push to remote (optional, good for backup)
git push origin feature/your-feature-name
```

### Finishing a feature

```bash
# Switch to dev and update it
git checkout dev
git pull origin dev

# Merge your feature
git merge feature/your-feature-name

# Delete the feature branch (optional)
git branch -d feature/your-feature-name

# Push to remote
git push origin dev
```

### Creating a release (merging to main)

```bash
# When dev is stable and ready for release
git checkout main
git pull origin main

# Merge dev into main
git merge dev

# Tag the release (optional but recommended)
git tag -a v1.0 -m "First release"

# Push to remote
git push origin main
git push origin --tags
```

## Commit Message Guidelines

Use clear, descriptive commit messages:

- `feat: add data preprocessing pipeline`
- `fix: correct missing value imputation`
- `docs: update README with usage instructions`
- `refactor: reorganize model training code`
- `test: add unit tests for preprocessing`
- `chore: update requirements.txt`


## Quick Commands

```bash
# Check which branch you're on
git branch

# See all branches
git branch -a

# Switch branches
git checkout branch-name

# See status
git status

# See commit history
git log --oneline --graph --all
```
