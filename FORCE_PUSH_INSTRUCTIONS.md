# Force Push Instructions

## The Problem
GitHub branch protection rules prevent force-pushing to main branch.

## ✓ Good News
The tokens have been **completely removed** from your local git history!

## Next Steps

### Option 1: Temporarily Disable Branch Protection (Recommended)

1. Go to: https://github.com/SerendipityOneInc/look-bench/settings/rules
2. Click on the rule that protects the main branch
3. Click "Edit" or the pencil icon
4. **Temporarily uncheck** "Do not allow force pushes"
5. Click "Save changes"
6. **Immediately run**: 
   ```bash
   cd /Users/siqiao/Documents/workarea/github/look-bench
   git push origin main --force
   git push origin gaochao-dev --force
   git push origin --tags --force
   ```
7. **Re-enable** branch protection after pushing
8. **Done!** The tokens are now removed from GitHub

### Option 2: Push to New Branch (Alternative)

If you can't modify branch protection:

```bash
cd /Users/siqiao/Documents/workarea/github/look-bench

# Push cleaned history to a new branch
git push origin main:clean-main --force

# On GitHub:
# 1. Go to Settings → Branches
# 2. Change default branch from 'main' to 'clean-main'
# 3. Delete the old 'main' branch
# 4. Rename 'clean-main' back to 'main'
# 5. Update default branch back to 'main'
```

### Option 3: Ask Repository Admin

If you're not the admin:
1. Ask the repository owner/admin to temporarily disable branch protection
2. They can do this at: https://github.com/SerendipityOneInc/look-bench/settings/rules
3. After you force push, they can re-enable it

## After Force Push

### 1. Verify on GitHub
- Go to: https://github.com/SerendipityOneInc/look-bench
- Check that the `examples/` folder is gone
- Check commit history

### 2. Dismiss Security Alerts
1. Go to: https://github.com/SerendipityOneInc/look-bench/security
2. Find the Dependabot alerts for the tokens
3. Click "Dismiss" on each alert
4. Select reason: "The secret is not valid" (since you revoked them)

### 3. Notify Collaborators (if any)
Anyone with a local clone needs to:
```bash
cd look-bench
git fetch origin
git reset --hard origin/main
git clean -fdx
```

Or just re-clone:
```bash
rm -rf look-bench
git clone https://github.com/SerendipityOneInc/look-bench.git
```

## Current Status

✅ Local repository cleaned (tokens removed from all history)
✅ Backup created: `backup-emergency-20260120` branch
✅ All notebook files removed from history
⏳ Waiting for force push to GitHub

## What Was Cleaned

Removed from all git history:
- `examples/01_data_structure.ipynb` (contained token)
- `examples/02_evaluation_metrics.ipynb`
- `examples/03_pipeline_flow.ipynb`

## Security Improvements Added

✅ `.gitignore` updated to exclude notebooks
✅ Security cleanup guides created
✅ Automated cleanup scripts added

## IMPORTANT REMINDERS

🔴 **Have you revoked the tokens on HuggingFace?**
   https://huggingface.co/settings/tokens

🔴 **After force push, dismiss GitHub security alerts**

🔴 **Re-enable branch protection after pushing**

## Need Help?

If you encounter issues:
```bash
# Check status
git status

# Check branches
git branch -a

# Check what's different from remote
git log origin/main..main --oneline
```

For questions or issues, refer to `SECURITY_CLEANUP.md` or `CLEANUP_STEPS.md`.
