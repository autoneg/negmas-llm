
# Repository: projects/negmas-llm
echo -e "${BLUE}Restoring: projects/negmas-llm${NC}"
if [ -d "projects/negmas-llm/.git" ]; then
    echo -e "  ${YELLOW}Directory already exists, skipping...${NC}"
else
    # Create parent directory if needed
    mkdir -p "projects"

    # Clone the repository
    if git clone "git@github.com:autoneg/negmas-llm.git" "projects/negmas-llm"; then
        echo -e "  ${GREEN}✓${NC} Successfully cloned"

        # Checkout the original branch if not already on it
        cd "projects/negmas-llm"
        current=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current" != "main" ]; then
            if git checkout "main" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Checked out branch: main"
            else
                echo -e "  ${YELLOW}⚠${NC} Could not checkout branch: main"
            fi
        fi
        cd - > /dev/null
    else
        echo -e "  ${RED}✗${NC} Failed to clone"
    fi
fi

# Repository: projects/negmas-llm
echo -e "${BLUE}Restoring: projects/negmas-llm${NC}"
if [ -d "projects/negmas-llm/.git" ]; then
    echo -e "  ${YELLOW}Directory already exists, skipping...${NC}"
else
    # Create parent directory if needed
    mkdir -p "projects"
    
    # Clone the repository
    if git clone "git@github.com:autoneg/negmas-llm.git" "projects/negmas-llm"; then
        echo -e "  ${GREEN}✓${NC} Successfully cloned"
        
        # Checkout the original branch if not already on it
        cd "projects/negmas-llm"
        current=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current" != "main" ]; then
            if git checkout "main" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Checked out branch: main"
            else
                echo -e "  ${YELLOW}⚠${NC} Could not checkout branch: main"
            fi
        fi
        cd - > /dev/null
    else
        echo -e "  ${RED}✗${NC} Failed to clone"
    fi
fi

