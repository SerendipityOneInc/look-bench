# Cleanup Execution Steps

## 🔴 STEP 1: Revoke Tokens (CRITICAL - Do this NOW!)

**This is the most important step!** Even if we clean the git history, the tokens are public and may have been accessed.

1. Open: https://huggingface.co/settings/tokens
2. Look for any tokens created around the time mentioned in the GitHub alert
3. **Delete them immediately**
4. Create new tokens if needed (never commit them!)

---

## STEP 2: Verify Current State

Run this to check if tokens exist locally:
```bash
cd /Users/siqiao/Documents/workarea/github/look-bench
git log --all --full-history -S "hf_" --oneline
```

---

## STEP 3: Clean Local Repository

Since the security alert came from GitHub, let's ensure everything is clean:

### Option A: Simple approach (if tokens are only on remote)
```bash
cd /Users/siqiao/Documents/workarea/github/look-bench

# Push current clean state
git push origin main --force
```

### Option B: Deep clean (if unsure)
```bash
cd /Users/siqiao/Documents/workarea/github/look-bench

# Create backup
git branch backup-$(date +%Y%m%d)

# Install BFG (much faster than filter-branch)
# On macOS:
brew install bfg

# Or download from: https://rtyley.github.io/bfg-repo-cleaner/

# Use BFG to remove any files that might contain tokens
bfg --delete-files '*.ipynb' --no-blob-protection
bfg --replace-text <(echo "hf_*==>REMOVED") --no-blob-protection

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Push
git push origin --force --all
git push origin --force --tags
```

### Option C: Using git filter-branch (if BFG not available)
```bash
cd /Users/siqiao/Documents/workarea/github/look-bench

# Create backup
git branch backup-$(date +%Y%m%d)

# Remove any .ipynb files from history
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch "*.ipynb"' \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Push
git push origin --force --all
git push origin --force --tags
```

---

## STEP 4: Verify Cleanup

```bash
# Check no tokens remain in history
cd /Users/siqiao/Documents/workarea/github/look-bench
git log --all --full-history -S "hf_" --oneline

# Should return empty!
```

---

## STEP 5: GitHub Security

After force pushing:

1. Go to your GitHub repository: https://github.com/SerendipityOneInc/look-bench
2. Go to Settings → Security → Dependabot alerts
3. If alerts still show, click "Dismiss" on each one
4. In the dismiss dialog, select "The secret is not valid" (since you revoked them)

---

## STEP 6: Prevent Future Issues

### A. Use environment variables

Create a `.env` file (already in .gitignore):
```bash
echo "HUGGINGFACE_TOKEN=your_new_token_here" > .env
```

In your code:
```python
import os
from dotenv import load_dotenv

load_dotenv()
token = os.environ.get('HUGGINGFACE_TOKEN')
```

### B. Install git-secrets

```bash
# Install
brew install git-secrets

# Setup for this repo
cd /Users/siqiao/Documents/workarea/github/look-bench
git secrets --install
git secrets --add 'hf_[a-zA-Z0-9]{30,}'
git secrets --add 'sk-[a-zA-Z0-9]{32,}'  # OpenAI keys
git secrets --add '[aA]PI[_-]?[kK]ey'    # Generic API keys
```

### C. Add pre-commit hook

Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
if git diff --cached | grep -i "hf_"; then
    echo "❌ ERROR: Potential HuggingFace token detected!"
    echo "Please remove tokens before committing."
    exit 1
fi
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

---

## Quick Command Summary

If you want to do everything at once:

```bash
cd /Users/siqiao/Documents/workarea/github/look-bench

# 1. Create backup
git branch backup-$(date +%Y%m%d)

# 2. Force push current clean state
git push origin main --force

# 3. Verify
git log --all -S "hf_" --oneline

# 4. If tokens still found, use BFG or filter-branch above
```

---

## ⚠️ Important Notes

- **Force pushing rewrites history** - anyone else with this repo needs to re-clone
- **Revoke tokens FIRST** - this is most critical
- **Don't skip verification** - make sure tokens are really gone
- **Update GitHub alerts** - dismiss them after cleanup

---

## Need Help?

If you encounter issues:
1. Check git status: `git status`
2. Check remotes: `git remote -v`
3. Check branches: `git branch -a`
4. Check history: `git log --oneline -10`
