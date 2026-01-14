#!/bin/bash
# Code formatting script for cal-ql-torch project

set -e

echo "🎨 Code Formatter for cal-ql-torch"
echo "================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create backup if requested
if [ "$1" = "--backup" ]; then
    BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}Creating backup in: $BACKUP_DIR${NC}"
    mkdir -p "$BACKUP_DIR"
    find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.yml" \) -exec cp --parents {} "$BACKUP_DIR" \;
    echo -e "${GREEN}✓ Backup created${NC}"
fi

# Format Python files
echo -e "${YELLOW}Formatting Python files...${NC}"
if command -v black &> /dev/null; then
    echo "Running black..."
    black .
    echo -e "${GREEN}✓ Black formatting complete${NC}"
else
    echo "Black not found. Install with: pip install black"
fi

if command -v isort &> /dev/null; then
    echo "Running isort..."
    isort .
    echo -e "${GREEN}✓ isort import sorting complete${NC}"
else
    echo "isort not found. Install with: pip install isort"
fi

# Format YAML files
echo -e "${YELLOW}Formatting YAML files...${NC}"
if command -v prettier &> /dev/null; then
    echo "Running prettier..."
    prettier --write "**/*.{yaml,yml}"
    echo -e "${GREEN}✓ Prettier formatting complete${NC}"
else
    echo "Using Python YAML formatter..."
    python -c "
import yaml
import glob
import os

for file in glob.glob('**/*.{yaml,yml}', recursive=True):
    if any(skip in file for skip in ['.git', '__pycache__', 'checkpoints', 'outputs', 'logs']):
        continue
    try:
        with open(file, 'r') as f:
            data = yaml.safe_load(f)
        with open(file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)
        print(f'Formatted: {file}')
    except Exception as e:
        print(f'Error formatting {file}: {e}')
"
    echo -e "${GREEN}✓ YAML formatting complete${NC}"
fi

# Check if there are any formatting issues
echo -e "${YELLOW}Checking formatting...${NC}"
if command -v black &> /dev/null; then
    if black --check . > /dev/null 2>&1; then
        echo -e "${GREEN}✓ All Python files properly formatted${NC}"
    else
        echo -e "${YELLOW}⚠ Some Python files still need formatting${NC}"
    fi
fi

echo -e "${GREEN}✨ Code formatting complete!${NC}"
