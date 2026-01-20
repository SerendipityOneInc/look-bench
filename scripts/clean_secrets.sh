#!/bin/bash
# Script to clean secrets from git history
# WARNING: This will rewrite git history!

set -e

echo "⚠️  WARNING: This will rewrite git history!"
echo "⚠️  Make sure you have:"
echo "   1. Revoked the leaked tokens on Hugging Face"
echo "   2. Backed up your repository"
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

cd "$(dirname "$0")/.."

echo "Creating backup branch..."
git branch backup-before-cleanup 2>/dev/null || echo "Backup branch already exists"

echo "Removing secrets from git history..."

# Method 1: Using git filter-branch (built-in)
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch examples/01_data_structure.ipynb' \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
echo "Cleaning up..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "✅ Secrets removed from git history!"
echo ""
echo "Next steps:"
echo "1. Verify the changes: git log --oneline"
echo "2. Force push to remote: git push origin --force --all"
echo "3. Force push tags: git push origin --force --tags"
echo ""
echo "⚠️  IMPORTANT: All collaborators must re-clone the repository!"
