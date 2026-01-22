# Token Cleanup Completion Checklist

## ✅ Completed Steps

- [x] Removed notebook files from local git history
- [x] Created backup branch (`backup-emergency-20260120`)
- [x] Pushed cleaned branch to GitHub as `main-cleaned`
- [x] Added `.gitignore` to prevent future leaks

## 🔄 In Progress - Complete These Steps

### Step 1: Change Default Branch (ON GITHUB)
- [ ] Go to: https://github.com/SerendipityOneInc/look-bench/settings/branches
- [ ] Click the ⇄ switch icon next to "Default branch"
- [ ] Select `main-cleaned`
- [ ] Click "Update" and confirm

### Step 2: Run Completion Script
After completing Step 1, run:
```bash
cd /Users/siqiao/Documents/workarea/github/look-bench
./finish_cleanup.sh
```

This will:
- Delete the old main branch
- Push cleaned version as new main
- Clean up temporary branches
- Update your local repository

### Step 3: Restore Default Branch Name (ON GITHUB)
- [ ] Go back to: https://github.com/SerendipityOneInc/look-bench/settings/branches
- [ ] Change default branch back to `main`
- [ ] Click "Update" and confirm

### Step 4: Verify Cleanup
- [ ] Visit: https://github.com/SerendipityOneInc/look-bench
- [ ] Confirm `examples/` folder is gone
- [ ] Check git history doesn't show notebook files

### Step 5: Dismiss Security Alerts
- [ ] Go to: https://github.com/SerendipityOneInc/look-bench/security
- [ ] Click on each Dependabot alert for the tokens
- [ ] Click "Dismiss alert"
- [ ] Select "The secret is not valid"
- [ ] Add comment: "Token revoked and removed from git history"

### Step 6: Verify Token Revocation
- [ ] Go to: https://huggingface.co/settings/tokens
- [ ] Confirm the leaked tokens are deleted:
  - `hf_haYPQXtcPLhqSEd0Eumov...`
  - `hf_xLlwqMbXKwCn0qukvKZjs...`
- [ ] Generate new tokens if needed (NEVER commit them!)

### Step 7: Notify Team (if applicable)
If others have cloned this repository:
- [ ] Notify collaborators about the history rewrite
- [ ] Ask them to re-clone:
  ```bash
  rm -rf look-bench
  git clone https://github.com/SerendipityOneInc/look-bench.git
  ```

## 📋 Quick Reference

### Current Branch Status
- `main` (local) - Cleaned version
- `main-cleaned` (remote) - Cleaned version on GitHub
- `origin/main` (remote) - OLD version with tokens
- `backup-emergency-20260120` - Backup before cleanup

### Key URLs
- Settings: https://github.com/SerendipityOneInc/look-bench/settings
- Security: https://github.com/SerendipityOneInc/look-bench/security
- Branches: https://github.com/SerendipityOneInc/look-bench/branches
- HuggingFace: https://huggingface.co/settings/tokens

### Verification Commands
```bash
# Check no tokens in history
git log --all -S "hf_" --oneline

# Check no notebooks in history
git log --all --name-only --pretty=format: | sort -u | grep -E "\.ipynb$"

# Check current status
git status

# Check branches
git branch -a
```

## ⚠️ Important Reminders

1. **Token revocation is CRITICAL** - Do this first!
2. **Don't skip dismissing GitHub alerts** - They'll keep showing otherwise
3. **Test with new tokens** - Use environment variables, not hard-coded values
4. **Update team** - If anyone has cloned, they need to re-clone

## 🎯 Success Criteria

Cleanup is complete when:
- [ ] No notebook files in git history
- [ ] GitHub security alerts dismissed
- [ ] HuggingFace tokens revoked
- [ ] Repository uses cleaned main branch
- [ ] `.gitignore` prevents future commits of notebooks
- [ ] Team notified (if applicable)

## 📞 Need Help?

If you encounter issues:
1. Check `ALTERNATIVE_SOLUTION.md` for different approaches
2. Check `FORCE_PUSH_INSTRUCTIONS.md` for detailed steps
3. Verify your git status: `git status`
4. Check branches: `git branch -a`

---

**Last Updated:** After pushing `main-cleaned` branch
**Status:** Awaiting default branch change on GitHub
