#!/bin/bash
set -e

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting GitHub Upload Process...${NC}"

# Navigate to the parent directory (EVK root)
# The script is expected to be in onnx_convert/
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}Project Root determined as: $PROJECT_ROOT${NC}"
cd "$PROJECT_ROOT"

# Check for nested git repo in onnx_convert and remove it if present
# to avoid it being treated as a submodule
if [ -d "$SCRIPT_DIR/.git" ]; then
    echo -e "${YELLOW}Found existing .git directory in onnx_convert. Removing it to merge into main repo...${NC}"
    rm -rf "$SCRIPT_DIR/.git"
fi

# Explicitly delete the folder as requested by user
if [ -d "iq9_ubuntu_images" ]; then
    echo -e "${YELLOW}Deleting iq9_ubuntu_images as requested...${NC}"
    rm -rf "iq9_ubuntu_images"
fi

# Configure Git Identity
echo -e "${GREEN}Configuring Git Identity...${NC}"
git config --global user.email "june2450@naver.com"
git config --global user.name "lemongzamong"

# Reset git to handle large files cleanup if it was partially added
if [ -d ".git" ]; then
    echo -e "${YELLOW}Removing existing .git directory to ensure fresh start with new ignore rules...${NC}"
    rm -rf ".git"
fi

echo -e "${GREEN}Initializing Git repository...${NC}"
git init
git branch -M main

# Create .gitignore
echo -e "${GREEN}Creating/Updating .gitignore...${NC}"
cat > .gitignore << EOL
__pycache__/
*.py[cod]
*$py.class
.env
.idea/
.vscode/
*.swp
.DS_Store

# Ignore virtual environments recursively
**/venv/
**/.venv/
env/

# Ignore large directories
iq9_ubuntu_images/
EOL

# Add files
echo -e "${GREEN}Adding files to git...${NC}"
git add .

# Commit
echo -e "${GREEN}Committing files...${NC}"
if git diff-index --quiet HEAD --; then
    echo "No changes to commit."
else
    git commit -m "Initial commit of IQ-9075 EVK project"
fi

# Check for Remote URL argument
REPO_URL="$1"

if [ -z "$REPO_URL" ]; then
    echo -e "${YELLOW}No GitHub URL provided as argument.${NC}"
    echo "Git repository initialized and files committed."
    echo "To push, run this script again with the URL: ./upload_to_github.sh <URL>"
    echo "Or provide the URL to the AI to finish the push."
    exit 0
fi

# Check if origin exists, if so check if it matches, else set it
if git remote get-url origin > /dev/null 2>&1; then
    EXISTING_URL=$(git remote get-url origin)
    if [ "$EXISTING_URL" != "$REPO_URL" ]; then
        echo -e "${YELLOW}Remote 'origin' already exists as $EXISTING_URL. Updating to $REPO_URL...${NC}"
        git remote set-url origin "$REPO_URL"
    else
        echo "Remote 'origin' already set correctly."
    fi
else
    git remote add origin "$REPO_URL"
fi

# Push
echo -e "${GREEN}Pushing to GitHub...${NC}"
git push -u origin main

echo -e "${GREEN}Upload Complete!${NC}"
