import streamlit as st
import subprocess
import os
import stat
import requests
import zipfile
import shutil
from urllib.parse import urljoin
from pathlib import Path
import sys
import pandas as pd

# RDKit imports for the standardize function
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# Import paths from paths.py
from .paths import (
    WORKSPACE_PARENT_DIR, LIGAND_PREP_DIR_LOCAL, VINA_PATH_LOCAL, VINA_DIR_LOCAL,
    VINA_EXECUTABLE_NAME
)

# --- Standardize Function ---
def standardize_smiles_rdkit(smiles, invalid_smiles_list):
    """Standardizes a SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles_list.append(smiles)
            return None

        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol)
        mol = Chem.RemoveHs(mol)

        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        mol = rdMolStandardize.Reionizer().reionize(mol)

        metal_disconnector = rdMolStandardize.MetalDisconnector()
        mol = metal_disconnector.Disconnect(mol)

        mol = rdMolStandardize.FragmentParent(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        standardized_smiles_out = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return standardized_smiles_out
    except Exception as e:
        st.warning(f"Error standardizing SMILES '{smiles}': {e}")
        invalid_smiles_list.append(smiles)
        return None

def initialize_directories():
    # Import all necessary directory paths from paths.py
    from .paths import (
        WORKSPACE_PARENT_DIR, RECEPTOR_DIR_LOCAL, CONFIG_DIR_LOCAL,
        LIGAND_PREP_DIR_LOCAL, LIGAND_UPLOAD_TEMP_DIR,
        ZIP_EXTRACT_DIR_LOCAL, DOCKING_OUTPUT_DIR_LOCAL,
        ENSEMBLE_DOCKING_DIR_LOCAL, LIGAND_PREPROCESSING_SUBDIR_LOCAL, VINA_DIR_LOCAL
    )
    dirs_to_create = [
        WORKSPACE_PARENT_DIR, RECEPTOR_DIR_LOCAL, CONFIG_DIR_LOCAL,
        LIGAND_PREP_DIR_LOCAL, LIGAND_UPLOAD_TEMP_DIR,
        ZIP_EXTRACT_DIR_LOCAL, DOCKING_OUTPUT_DIR_LOCAL,
        ENSEMBLE_DOCKING_DIR_LOCAL, LIGAND_PREPROCESSING_SUBDIR_LOCAL, VINA_DIR_LOCAL
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

def list_files_from_github_repo_dir(owner: str, repo: str, dir_path_in_repo: str, branch: str, gh_api_base_url: str, file_extension: str = None) -> list[str]:
    api_url = f"{gh_api_base_url}{owner}/{repo}/contents/{dir_path_in_repo}?ref={branch}"
    filenames = []
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        contents = response.json()
        if not isinstance(contents, list):
            st.sidebar.error(f"API Error for {dir_path_in_repo}: Expected list.")
            if isinstance(contents, dict) and 'message' in contents: st.sidebar.error(f"GitHub: {contents['message']}")
            return []
        for item in contents:
            if item.get('type') == 'file':
                if file_extension:
                    if item.get('name', '').lower().endswith(file_extension.lower()):
                        filenames.append(item['name'])
                else:
                    filenames.append(item['name'])
        if not filenames and file_extension:
            st.sidebar.caption(f"No files matching '{file_extension}' found in '{dir_path_in_repo}'.")
    except Exception as e:
        st.sidebar.error(f"Error listing files from GitHub ({dir_path_in_repo}): {e}")
    return filenames

def download_file_from_github(raw_download_base_url, relative_path_segment, local_filename, local_save_dir):
    full_url = urljoin(raw_download_base_url, relative_path_segment)
    local_file_path = Path(local_save_dir) / local_filename
    try:
        response = requests.get(full_url, stream=True, timeout=15)
        response.raise_for_status()
        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        return str(local_file_path)
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error downloading {local_filename} from {full_url}: {e}")
        return None

def make_file_executable(filepath_str):
    if not filepath_str or not os.path.exists(filepath_str):
        return False
    try:
        os.chmod(filepath_str, os.stat(filepath_str).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to make {filepath_str} executable: {e}")
        return False

def check_script_exists(script_path: Path, script_name: str, is_critical: bool = True):
    if script_path.exists() and script_path.is_file(): return True
    msg_func = st.sidebar.error if is_critical else st.sidebar.warning
    msg_func(f"{'CRITICAL: ' if is_critical else ''}`{script_name}` NOT FOUND at `{script_path}`.")
    return False

def check_vina_binary(show_success=True):
    if not VINA_PATH_LOCAL.exists():
        st.sidebar.error(f"Vina exe NOT FOUND at `{VINA_PATH_LOCAL}`. Ensure `{VINA_EXECUTABLE_NAME}` is in `{VINA_DIR_LOCAL}`.")
        return False
    if show_success: st.sidebar.success(f"Vina binary found: {VINA_PATH_LOCAL.name}")

    if os.access(str(VINA_PATH_LOCAL.resolve()), os.X_OK):
        if show_success: st.sidebar.success("Vina binary is executable.")
        return True
    else:
        st.sidebar.warning("Vina binary NOT executable by os.access. Attempting permission set...")
        if make_file_executable(str(VINA_PATH_LOCAL)):
            if os.access(str(VINA_PATH_LOCAL.resolve()), os.X_OK):
                st.sidebar.success("Execute permission successfully set for Vina and verified.")
                return True
            else:
                st.sidebar.error("Failed to make Vina executable (os.access still fails after chmod).")
                st.sidebar.markdown(f"**Manual Action Needed:** `git add --chmod=+x {VINA_DIR_LOCAL.name}/{VINA_EXECUTABLE_NAME}` in your local repo, commit, and push. Or ensure the file has execute permissions in your deployment environment.")
                return False
        else:
            st.sidebar.error("Failed to make Vina executable (chmod call failed).")
            return False

def get_smiles_from_pubchem_inchikey(inchikey_str):
    api_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey_str}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status(); data = response.json()
        return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
    except Exception as e: st.warning(f"PubChem API/parse error for {inchikey_str}: {e}"); return None

def run_ligand_prep_script(script_local_path_str, script_args, process_name, ligand_name_for_log):
    if not script_local_path_str: st.error(f"{process_name}: Script path undefined."); return False
    absolute_script_path = str(Path(script_local_path_str).resolve())
    if not os.path.exists(absolute_script_path):
        st.error(f"{process_name} script NOT FOUND: {absolute_script_path}"); return False

    if not os.access(absolute_script_path, os.X_OK):
        if not make_file_executable(absolute_script_path) or not os.access(absolute_script_path, os.X_OK):
            st.error(f"Failed to make {process_name} script executable. Cannot run.")
            return False

    command = [sys.executable, absolute_script_path] + [str(arg) for arg in script_args]
    cwd_path_resolved = str(WORKSPACE_PARENT_DIR.resolve())
    if not os.path.exists(cwd_path_resolved):
        st.error(f"Working directory {cwd_path_resolved} for {process_name} missing."); return False
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=cwd_path_resolved)
        if result.stdout.strip():
            with st.expander(f"{process_name} STDOUT for {ligand_name_for_log}", expanded=False): st.text(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error during {process_name} for {ligand_name_for_log} (RC: {e.returncode}):")
        with st.expander(f"{process_name} Details (on error)", expanded=True):
            st.error(f"Command: `{' '.join(e.cmd)}`")
            st.text("STDOUT:\n" + (e.stdout.strip() or "No STDOUT."))
            st.text("STDERR:\n" + (e.stderr.strip() or "No STDERR."))
        return False
    except Exception as e: st.error(f"Unexpected error running {process_name} for {ligand_name_for_log}: {e}"); return False

def convert_smiles_to_pdbqt(smiles_str, ligand_name_base, output_dir_path_for_final_pdbqt, ph_val, skip_taut, skip_acidbase, local_scrub_script_path, local_mk_prepare_script_path):
    output_dir_path_for_final_pdbqt.mkdir(parents=True, exist_ok=True)
    # Use .name attribute for constructing relative paths within WORKSPACE_PARENT_DIR
    relative_sdf_filename = Path(LIGAND_PREP_DIR_LOCAL.name) / f"{ligand_name_base}_scrubbed.sdf"
    relative_pdbqt_filename = Path(LIGAND_PREP_DIR_LOCAL.name) / f"{ligand_name_base}.pdbqt"

    absolute_sdf_path_for_check = WORKSPACE_PARENT_DIR / relative_sdf_filename
    absolute_pdbqt_path_for_return = WORKSPACE_PARENT_DIR / relative_pdbqt_filename

    scrub_options = ["--ph", str(ph_val)]
    if skip_taut: scrub_options.append("--skip_tautomer")
    if skip_acidbase: scrub_options.append("--skip_acidbase")
    scrub_args = [smiles_str, "-o", str(relative_sdf_filename)] + scrub_options

    if not run_ligand_prep_script(str(local_scrub_script_path), scrub_args, "scrub.py", ligand_name_base): return None
    if not absolute_sdf_path_for_check.exists():
        st.error(f"scrub.py did not produce expected output: {absolute_sdf_path_for_check}")
        return None

    mk_prepare_args = ["-i", str(relative_sdf_filename), "-o", str(relative_pdbqt_filename)]
    if not run_ligand_prep_script(str(local_mk_prepare_script_path), mk_prepare_args, "mk_prepare_ligand.py", ligand_name_base): return None

    return {"id": smiles_str, "pdbqt_path": str(absolute_pdbqt_path_for_return), "base_name": ligand_name_base} if absolute_pdbqt_path_for_return.exists() else None

def convert_ligand_file_to_pdbqt(input_ligand_file_path_absolute, original_filename, output_dir_path_for_final_pdbqt, local_mk_prepare_script_path):
    output_dir_path_for_final_pdbqt.mkdir(parents=True, exist_ok=True)
    ligand_name_base = Path(original_filename).stem
    # Use .name attribute for constructing relative paths within WORKSPACE_PARENT_DIR
    relative_pdbqt_filename = Path(LIGAND_PREP_DIR_LOCAL.name) / f"{ligand_name_base}.pdbqt"
    absolute_pdbqt_path_for_return = WORKSPACE_PARENT_DIR / relative_pdbqt_filename

    mk_prepare_args = ["-i", str(Path(input_ligand_file_path_absolute).resolve()), "-o", str(relative_pdbqt_filename)]

    if not run_ligand_prep_script(str(local_mk_prepare_script_path), mk_prepare_args, "mk_prepare_ligand.py", ligand_name_base): return None
    return {"id": original_filename, "pdbqt_path": str(absolute_pdbqt_path_for_return), "base_name": ligand_name_base} if absolute_pdbqt_path_for_return.exists() else None

def find_paired_config_for_protein(protein_base_name: str, all_config_paths: list[str]) -> Path | None:
    if not all_config_paths: return None
    patterns_to_try = [f"{protein_base_name}.txt", f"config_{protein_base_name}.txt", f"{protein_base_name}_config.txt"]
    for pattern in patterns_to_try:
        for cfg_path_str in all_config_paths:
            cfg_file = Path(cfg_path_str)
            if cfg_file.name.lower() == pattern.lower(): return cfg_file
    for cfg_path_str in all_config_paths:
        cfg_file = Path(cfg_path_str)
        if cfg_file.suffix.lower() == ".txt" and protein_base_name.lower() in cfg_file.stem.lower():
            if "config" in cfg_file.stem.lower() or cfg_file.stem.lower() == protein_base_name.lower(): return cfg_file
    return None

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def parse_score_from_pdbqt(pdbqt_file_path: str) -> float | None:
    try:
        resolved_path = Path(pdbqt_file_path).resolve()
        if not resolved_path.exists():
            st.warning(f"PDBQT file for score parsing not found: {resolved_path}")
            return None
        if resolved_path.stat().st_size == 0:
            st.warning(f"PDBQT file for score parsing is empty: {resolved_path}")
            return None
        with open(resolved_path, 'r', encoding='utf-8') as f: lines = f.readlines()
        for i, line_content in enumerate(lines):
            current_line_stripped = line_content.strip()
            if current_line_stripped.upper().startswith("REMARK VINA RESULT:"):
                parts = current_line_stripped.split(':', 1)
                if len(parts) > 1:
                    score_values_str = parts[1].strip()
                    score_values_list = score_values_str.split()
                    if score_values_list:
                        try:
                            score = float(score_values_list[0])
                            return score
                        except ValueError:
                            st.warning(f"ValueError converting score part '{score_values_list[0]}' to float in {resolved_path.name} on line {i+1}.")
                            return None
                    else:
                        st.warning(f"Score part list is empty after split for VINA RESULT line in {resolved_path.name}.")
                        return None
                else:
                    st.warning(f"'REMARK VINA RESULT:' found, but line splitting by ':' failed in {resolved_path.name}.")
                    return None
        st.warning(f"Could not find 'REMARK VINA RESULT:' in {resolved_path.name}.")
        return None
    except Exception as e:
        st.error(f"Unexpected error parsing PDBQT file {Path(pdbqt_file_path).name}: {e}")
        return None
