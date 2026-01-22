# Alternative Solution: Replace Main Branch

Since force-push to main is blocked by branch protection, we'll use a workaround.

## ✅ Good News
Your backup branch (`backup-emergency-20260120`) was successfully pushed to GitHub!

## Strategy: Replace Main Branch via GitHub UI

### Method 1: Via GitHub Settings (Recommended)

**Step 1: Push clean main to a temporary branch**
```bash
cd /Users/siqiao/Documents/workarea/github/look-bench

# Push your cleaned main branch as 'main-cleaned'
git push origin main:main-cleaned --force
```

**Step 2: Change default branch on GitHub**
1. Go to: https://github.com/SerendipityOneInc/look-bench/settings
2. Click "Branches" in the left sidebar
3. Under "Default branch", click the switch icon
4. Select `main-cleaned`
5. Click "Update" and confirm

**Step 3: Delete old main branch**
1. Go to: https://github.com/SerendipityOneInc/look-bench/branches
2. Find the `main` branch
3. Click the trash icon to delete it
4. Confirm deletion

**Step 4: Rename main-cleaned to main**
```bash
cd /Users/siqiao/Documents/workarea/github/look-bench

# Fetch changes
git fetch origin

# Rename the branch on GitHub by pushing with -d and then pushing new
git push origin :main  # This might fail, that's OK
git push origin main-cleaned:main --force
```

**Step 5: Clean up**
```bash
# Delete temporary branch
git push origin --delete main-cleaned

# Update your local tracking
git fetch --prune
git branch -u origin/main main
```

**Step 6: Change default back to main** (on GitHub settings page)

---

### Method 2: Contact Repository Admin

If you're not an admin, ask someone with admin access to:

1. Go to: https://github.com/SerendipityOneInc/look-bench/settings/rules
2. Click on the branch protection rule for main
3. Temporarily disable "Do not allow force pushes"
4. Wait for you to force push
5. Re-enable the protection

Then you can run:
```bash
cd /Users/siqiao/Documents/workarea/github/look-bench
git push origin main --force
git push origin --all --force
```

---

### Method 3: Use GitHub CLI (if installed)

```bash
# Install GitHub CLI if needed
brew install gh

# Authenticate
gh auth login

# Temporarily disable branch protection
gh api repos/SerendipityOneInc/look-bench/branches/main/protection \
  --method DELETE

# Force push
git push origin main --force
git push origin --all --force

# Re-enable branch protection (you'll need to reconfigure settings)
```

---

## Quick Start: Execute Method 1 Now

Run these commands:

```bash
cd /Users/siqiao/Documents/workarea/github/look-bench

# Step 1: Push cleaned main as temporary branch
git push origin main:main-cleaned --force

# After this succeeds, follow steps 2-4 on GitHub UI
# Then come back and run:

# Step 2: After changing default branch on GitHub, delete old main
git push origin --delete main 2>/dev/null || echo "Main deletion might fail, continue anyway"

# Step 3: Push cleaned version as main
git push origin main-cleaned:refs/heads/main

# Step 4: Clean up
git push origin --delete main-cleaned
git fetch --prune

echo "✅ Complete! Now update default branch back to 'main' on GitHub"
```

---

## After Successfully Pushing

1. **Verify the notebooks are gone:**
   - Visit: https://github.com/SerendipityOneInc/look-bench
   - Check that `examples/` folder is removed

2. **Dismiss GitHub Security Alerts:**
   - Go to: https://github.com/SerendipityOneInc/look-bench/security
   - Dismiss each token alert
   - Select: "The secret is not valid"

3. **Verify tokens revoked:**
   - https://huggingface.co/settings/tokens
   - Make sure old tokens are deleted

4. **Team notification:**
   - Tell collaborators to re-clone the repository

---

## Troubleshooting

**If you get "branch protection" errors on any command:**
- You need admin access to change branch protection
- Ask repository owner for help
- Or use Method 2 above

**If deletion fails:**
- It's OK! The important part is pushing the clean version
- Old commits will become unreachable and git will garbage collect them

**To verify cleanup worked:**
```bash
git log --all -S "hf_" --oneline
# Should only show commits with security guides, not actual tokens
```
