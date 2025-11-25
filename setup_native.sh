#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== C.O.R.A.L. Native Environment Setup ===${NC}"

# Function to setup conda environment
setup_conda_env() {
    ENV_NAME="coral-env"
    if conda info --envs | grep -q "$ENV_NAME"; then
        echo -e "${YELLOW}Environment '$ENV_NAME' already exists. Updating...${NC}"
        conda env update -f environment.yaml --prune
    else
        echo -e "${YELLOW}Creating new environment '$ENV_NAME'...${NC}"
        conda env create -f environment.yaml
    fi
    
    echo -e "\n${GREEN}=== Setup Complete! ===${NC}"
    echo -e "To start training:"
    
    if [ "$INSTALLED_CONDA" = true ]; then
        echo -e "${YELLOW}IMPORTANT: Since you just installed Conda, you need to initialize your shell first:${NC}"
        if [[ "$SHELL" == *"zsh"* ]]; then
            echo -e "   ${YELLOW}source ~/.zshrc${NC}"
        elif [[ "$SHELL" == *"bash"* ]]; then
             echo -e "   ${YELLOW}source ~/.bashrc${NC}"
        else
             echo -e "   ${YELLOW}source <your_shell_profile>${NC}"
        fi
        echo -e "Then activate the environment:"
    else
        echo -e "1. Activate the environment:"
    fi
    
    echo -e "   ${YELLOW}conda activate $ENV_NAME${NC}"
    echo -e "2. Run the training script:"
    echo -e "   ${YELLOW}python coral_o_former.py${NC}"
}

INSTALLED_CONDA=false

# 1. Check for Conda
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}Conda not found.${NC}"
    read -p "Would you like to install Miniconda? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Check if already installed but not in path
        if [ -d "$HOME/miniconda" ]; then
             echo -e "${YELLOW}Miniconda directory found at $HOME/miniconda${NC}"
             echo -e "${YELLOW}Skipping download/install and running initialization...${NC}"
             
             # Initialize conda for current shell
             eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
             
             # Initialize for both Bash and Zsh
             "$HOME/miniconda/bin/conda" init bash
             "$HOME/miniconda/bin/conda" init zsh
             
             INSTALLED_CONDA=true
             echo -e "${GREEN}✓ Conda initialized${NC}"
        else
            echo -e "${YELLOW}Installing Miniconda...${NC}"
            
            # Detect OS and Arch
            OS="$(uname)"
            ARCH="$(uname -m)"
            MINICONDA_URL=""
            
            if [ "$OS" = "Linux" ]; then
                if [ "$ARCH" = "x86_64" ]; then
                    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
                elif [ "$ARCH" = "aarch64" ]; then
                    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
                fi
            elif [ "$OS" = "Darwin" ]; then
                if [ "$ARCH" = "x86_64" ]; then
                    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
                elif [ "$ARCH" = "arm64" ]; then
                    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
                fi
            fi
            
            if [ -z "$MINICONDA_URL" ]; then
                echo -e "${RED}Could not detect compatible Miniconda version for $OS $ARCH${NC}"
            else
                echo -e "${YELLOW}Downloading Miniconda from $MINICONDA_URL...${NC}"
                curl -L -o miniconda.sh "$MINICONDA_URL"
                
                echo -e "${YELLOW}Running installer...${NC}"
                bash miniconda.sh -b -p "$HOME/miniconda"
                rm miniconda.sh
                
                # Initialize conda for current shell
                echo -e "${YELLOW}Initializing Conda...${NC}"
                eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
                
                # Initialize for both Bash and Zsh
                "$HOME/miniconda/bin/conda" init bash
                "$HOME/miniconda/bin/conda" init zsh
                
                INSTALLED_CONDA=true
                echo -e "${GREEN}✓ Miniconda installed successfully${NC}"
            fi
        fi
    fi
fi

# 2. Run Setup (Conda or Venv)
if command -v conda &> /dev/null; then
    echo -e "${GREEN}✓ Conda found at $(which conda)${NC}"
    setup_conda_env

elif command -v python3 &> /dev/null; then
    echo -e "${GREEN}✓ Python 3 found at $(which python3)${NC}"
    echo -e "${YELLOW}Conda not found. Falling back to venv...${NC}"
    
    VENV_DIR="coral-venv"
    
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}Creating virtual environment in '$VENV_DIR'...${NC}"
        python3 -m venv $VENV_DIR
    else
        echo -e "${YELLOW}Virtual environment '$VENV_DIR' already exists.${NC}"
    fi
    
    # Activate venv
    source $VENV_DIR/bin/activate
    
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install --upgrade pip
    
    # Install PyTorch with MPS support (default on Mac usually works, but explicit is good)
    # For Mac, standard pip install torch torchvision torchaudio works for MPS
    pip install torch torchvision torchaudio numpy matplotlib fastapi uvicorn
    
    echo -e "\n${GREEN}=== Setup Complete! ===${NC}"
    echo -e "To start training on your Mac (with GPU support):"
    echo -e "1. Activate the environment:"
    echo -e "   ${YELLOW}source $VENV_DIR/bin/activate${NC}"

else
    echo -e "${RED}Error: Neither Conda nor Python 3 found.${NC}"
    echo "Please install Python 3 or Miniconda."
    exit 1
fi

# 3. Instructions
echo -e "2. Run the training script:"
echo -e "   ${YELLOW}python coral_o_former.py${NC}"
