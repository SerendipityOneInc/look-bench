# Security Cleanup Guide

## ⚠️ URGENT: Leaked Tokens Detected

Your repository has leaked Hugging Face tokens in the git history. Follow these steps **immediately**:

## Step 1: Revoke the Leaked Tokens (DO THIS FIRST!)

1. Go to https://huggingface.co/settings/tokens
2. Find and **DELETE** the leaked tokens shown in the GitHub security alert
3. Generate new tokens if needed (never commit them to git!)

## Step 2: Clean Git History

### Option A: Using the provided script (Recommended)

```bash
cd /Users/siqiao/Documents/workarea/github/look-bench
./scripts/clean_secrets.sh
```

### Option B: Manual cleanup

```bash
cd /Users/siqiao/Documents/workarea/github/look-bench

# Create backup
git branch backup-before-cleanup

# Remove the notebook file from history (if it exists)
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch examples/01_data_structure.ipynb' \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Option C: Using BFG Repo-Cleaner (Fastest)

```bash
# Install BFG (if not installed)
brew install bfg  # macOS
# or download from https://rtyley.github.io/bfg-repo-cleaner/

# Clone a fresh copy
cd ..
git clone --mirror https://github.com/SerendipityOneInc/look-bench.git look-bench-mirror
cd look-bench-mirror

# Remove the file
bfg --delete-files 01_data_structure.ipynb

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Push
git push --force
```

## Step 3: Force Push to Remote

⚠️ **WARNING**: This will rewrite history on GitHub!

```bash
# Push all branches
git push origin --force --all

# Push all tags
git push origin --force --tags
```

## Step 4: Notify Collaborators

If anyone else has cloned this repository, they need to:

```bash
# Delete their local copy
rm -rf look-bench

# Re-clone
git clone https://github.com/SerendipityOneInc/look-bench.git
```

## Step 5: Prevent Future Leaks

### Add .gitignore

Already added! The `.gitignore` file now includes:
- `*.ipynb` - Jupyter notebooks
- `*.env` - Environment files
- `secrets.yaml` - Secret files
- `*_token.txt` - Token files

### Use Environment Variables

Instead of hardcoding tokens, use environment variables:

```python
import os
token = os.environ.get('HUGGINGFACE_TOKEN')
```

### Use git-secrets

Install git-secrets to prevent committing secrets:

```bash
# Install
brew install git-secrets  # macOS

# Setup for this repo
cd /Users/siqiao/Documents/workarea/github/look-bench
git secrets --install
git secrets --register-aws
git secrets --add 'hf_[a-zA-Z0-9]{30,}'  # Hugging Face tokens
```

## Verification

After cleanup, verify the tokens are gone:

```bash
# Search for tokens in history
git log --all --full-history --source --pretty=format:'%H' -S 'hf_haYPQXtcPLhqSEd0Eumov'
git log --all --full-history --source --pretty=format:'%H' -S 'hf_xLlwqMbXKwCn0qukvKZjs'

# Should return nothing if successful
```

## Resources

- [GitHub: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)
- [git-secrets](https://github.com/awslabs/git-secrets)
