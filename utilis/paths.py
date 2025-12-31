from pathlib import Path

APP_VERSION = "1.0.0" # Updated version

BASE_GITHUB_URL_FOR_DATA = "https://raw.githubusercontent.com/HenryChritopher02/GSJ/main/"
GH_API_BASE_URL = "https://api.github.com/repos/"
GH_OWNER = "HenryChritopher02"
GH_REPO = "GSJ"
GH_BRANCH = "main"
GH_ENSEMBLE_DOCKING_ROOT_PATH = "ensemble-docking"
RECEPTOR_SUBDIR_GH = "ensemble_protein/"
CONFIG_SUBDIR_GH = "config/"

APP_ROOT = Path(".") # Assumes streamlit_app.py is in the root of your project
ENSEMBLE_DOCKING_DIR_LOCAL = APP_ROOT / "utils"
LIGAND_PREPROCESSING_SUBDIR_LOCAL = ENSEMBLE_DOCKING_DIR_LOCAL / "ligand_preprocessing"
SCRUB_PY_LOCAL_PATH = LIGAND_PREPROCESSING_SUBDIR_LOCAL / "scrub.py"
MK_PREPARE_LIGAND_PY_LOCAL_PATH = LIGAND_PREPROCESSING_SUBDIR_LOCAL / "mk_prepare_ligand.py"
VINA_SCREENING_PL_LOCAL_PATH = ENSEMBLE_DOCKING_DIR_LOCAL / "Vina_screening.pl"

VINA_DIR_LOCAL = APP_ROOT / "vina"
VINA_EXECUTABLE_NAME = "vina_1.2.7_linux_x86_64" # Ensure this matches your Vina executable
VINA_PATH_LOCAL = VINA_DIR_LOCAL / VINA_EXECUTABLE_NAME

WORKSPACE_PARENT_DIR = APP_ROOT / "autodock_workspace"
RECEPTOR_DIR_LOCAL = WORKSPACE_PARENT_DIR / "fetched_receptors"
CONFIG_DIR_LOCAL = WORKSPACE_PARENT_DIR / "fetched_configs"
LIGAND_PREP_DIR_LOCAL = WORKSPACE_PARENT_DIR / "prepared_ligands"
LIGAND_UPLOAD_TEMP_DIR = WORKSPACE_PARENT_DIR / "uploaded_ligands_temp"
ZIP_EXTRACT_DIR_LOCAL = WORKSPACE_PARENT_DIR / "zip_extracted_ligands"
DOCKING_OUTPUT_DIR_LOCAL = APP_ROOT / "autodock_outputs"



