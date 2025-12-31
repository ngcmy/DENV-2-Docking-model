import re
import streamlit as st
import subprocess
import os
import zipfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from streamlit_ketcher import st_ketcher # For drawing molecules
import plotly.express as px
import py3Dmol
from stmol import showmol
from meeko import MoleculePreparation, PDBQTMolecule # FIX: Added PDBQTMolecule for reading

# Giữ lại các import từ file utils cục bộ để tận dụng cấu trúc hiện có
# Lưu ý: Vì logic đơn giản hóa, ta sẽ không dùng hết tất cả biến, nhưng giữ lại import để tránh lỗi
from utils.paths import (
    APP_VERSION, BASE_GITHUB_URL_FOR_DATA, 
    APP_ROOT, VINA_EXECUTABLE_NAME, VINA_PATH_LOCAL,
    RECEPTOR_DIR_LOCAL, CONFIG_DIR_LOCAL,
    LIGAND_PREP_DIR_LOCAL, LIGAND_UPLOAD_TEMP_DIR, ZIP_EXTRACT_DIR_LOCAL,
    DOCKING_OUTPUT_DIR_LOCAL, WORKSPACE_PARENT_DIR,
    SCRUB_PY_LOCAL_PATH, MK_PREPARE_LIGAND_PY_LOCAL_PATH
)
from utils.app_utils import (
    initialize_directories, download_file_from_github, 
    check_vina_binary, convert_df_to_csv,
    standardize_smiles_rdkit, convert_smiles_to_pdbqt
)

# --- CẤU HÌNH CÁC MỤC TIÊU TIỂU ĐƯỜNG ---
# Giả định các file này nằm trong thư mục 'receptors' và 'configs' trên GitHub
# Bạn cần đảm bảo tên file trên GitHub khớp với định nghĩa ở đây.
DIABETES_TARGETS = {
    "DPP-4 (4A5S)": {
        "pdbqt": "dpp4.pdbqt",
        "config": "dpp4.txt"
    },
    "GLP1-R (6X19)": {
        "pdbqt": "glp1r.pdbqt",
        "config": "glp1r.txt"
    },
    "PPAR-γ (5Y2O)": {
        "pdbqt": "pparg.pdbqt",
        "config": "pparg.txt"
    },
    "SGLT2 (8HEZ)": {
        "pdbqt": "sglt2.pdbqt",
        "config": "sglt2.txt"
    },
    "SUR1 (7S5V)": {
        "pdbqt": "sur1.pdbqt",
        "config": "sur1.txt"
    }
}

# Define ML Models (Ensure these exist in your GitHub 'models/' folder)
ML_MODELS_CONFIG = {
    "DPP-4": "dppiv.pkl",
    "PPAR-γ": "pparg.pkl",
    "GLP1-R": "glp1r.pkl"
}

MODELS_DIR_LOCAL = APP_ROOT / "models"

def load_ml_model(target_name):
    """Downloads and loads the .pkl model for the specific target."""
    MODELS_DIR_LOCAL.mkdir(parents=True, exist_ok=True)
    model_filename = ML_MODELS_CONFIG.get(target_name)
    if not model_filename:
        return None
    
    local_path = MODELS_DIR_LOCAL / model_filename
    
    # Download if not exists
    if not local_path.exists():
        with st.spinner(f"Downloading model for {target_name}..."):
            # Assuming models are in a 'models' folder in the repo
            download_file_from_github(BASE_GITHUB_URL_FOR_DATA, f"models/{model_filename}", model_filename, MODELS_DIR_LOCAL)
    
    if local_path.exists():
        try:
            return joblib.load(local_path)
        except Exception as e:
            st.error(f"Error loading model {model_filename}: {e}")
            return None
    return None

def calculate_ecfp4(smiles):
    """Calculates ECFP4 fingerprint (2048 bits) from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            return np.array(fp)
    except:
        return None
    return None

def convert_pdbqt_to_pdb(pdbqt_path, output_pdb_path):
    """
    Extracts the first pose from a PDBQT file and converts it to PDB format
    by stripping AutoDock-specific columns (Charge/AtomType).
    """
    try:
        with open(pdbqt_path, 'r') as f:
            lines = f.readlines()
        
        pdb_lines = []
        in_model = False
        model_found = False
        
        for line in lines:
            # Handle Models: Only take the first one
            if line.startswith("MODEL"):
                in_model = True
                model_found = True
                continue
            if line.startswith("ENDMDL"):
                break 
            
            # Process ATOM/HETATM lines
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # PDBQT format extends PDB columns. 
                # Standard PDB lines are usually ~80 chars. 
                # AutoDock adds charge/type info after column 66.
                # We slice up to column 66 to make it a valid PDB line, 
                # preserving coordinates (30-54) and occupancy/b-factor if present.
                # We can also explicitly reset Occupancy(55-60) and B-Factor(61-66) to 1.00/0.00 if needed,
                # but simple slicing is usually sufficient for visualization.
                clean_line = line[:66] + "\n"
                pdb_lines.append(clean_line)
            elif not model_found and not line.startswith("TORSDOF"): 
                # Keep header lines if they exist before MODEL
                pdb_lines.append(line)

        # Write to PDB file
        with open(output_pdb_path, 'w') as f:
            f.writelines(pdb_lines)
            
        return True
            
    except Exception as e:
        print(f"PDB Conversion Error: {e}")
        return False

def parse_vina_score_from_file(file_path):
    """
    Hàm đọc file output PDBQT và lấy điểm năng lượng liên kết thấp nhất (best affinity).
    """
    best_affinity = None
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('REMARK VINA RESULT'):
                    parts = line.split()
                    # Định dạng thường là: REMARK VINA RESULT: -9.5 0.000 0.000
                    if len(parts) >= 4:
                        best_affinity = float(parts[3])
                    break
    except Exception:
        pass
    return best_affinity

def run_single_docking(vina_path, receptor_path, ligand_path, config_path, output_path):
    """
    Hàm chạy Vina cho 1 cặp Receptor - Ligand.
    """
    cmd = [
        str(vina_path),
        "--receptor", str(receptor_path),
        "--ligand", str(ligand_path),
        "--config", str(config_path),
        "--out", str(output_path),
        "--cpu", "2" # Sử dụng 2 CPU cho mỗi tác vụ để cân bằng
    ]
    
    # Chạy lệnh
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr
    
def view_complex(protein_path, ligand_path):
    """
    Generates a 3D visualization. Detects format based on extension.
    """
    try:
        with open(protein_path, 'r') as f: protein_data = f.read()
        with open(ligand_path, 'r') as f: ligand_data = f.read()
        
        # Detect ligand format (pdbqt, sdf, mol2)
        ligand_format = Path(ligand_path).suffix.strip('.').lower()

        view = py3Dmol.view(width=800, height=500)
        view.addModelsAsFrames(protein_data)
        view.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}})
        
        # Add ligand with correct format
        view.addModel(ligand_data, ligand_format)
        view.setStyle({'model': -1}, {"stick": {'colorscheme': 'greenCarbon'}})
        
        view.zoomTo()
        showmol(view, height=500, width=800)
    except FileNotFoundError:
        st.error("Could not find files for visualization.")

def display_ml_prediction_procedure():
    st.header("🔮 Machine Learning Activity Prediction")
    st.info("Predict bioactivity (Active/Inactive) against DPPIV, PPARG, and GLP-1R using Machine Learning Models trained on ECFP4 fingerprints.")
    
    

    # 1. Select Targets
    st.subheader("1. Select Targets")
    selected_ml_targets = st.multiselect("Choose Target(s):", list(ML_MODELS_CONFIG.keys()), default=["DPP-4"])
    
    # 2. Input SMILES
    st.subheader("2. Input Molecules")
    input_type = st.radio("Input Method:", ["Enter SMILES", "Upload File (.txt)", "Draw Molecule", "Use Example"], horizontal=True)
    
    smiles_list = []
    
    if input_type == "Enter SMILES":
        text_in = st.text_area("Enter SMILES (one per line):")
        if text_in:
            smiles_list = [s.strip() for s in text_in.split('\n') if s.strip()]
            
    elif input_type == "Upload File (.txt)":
        up_file = st.file_uploader("Upload .txt (one SMILES per line)", type=['txt'])
        if up_file:
            stringio = up_file.getvalue().decode("utf-8")
            smiles_list = [s.strip() for s in stringio.split('\n') if s.strip()]
            
    elif input_type == "Draw Molecule":
        smile_art = st_ketcher(key="ml_ketcher")
        if smile_art:
            smiles_list = [smile_art]
            st.write(f"SMILES: `{smile_art}`")

    elif input_type == "Use Example":
        st.markdown("Using **Sitagliptin** (DPP-4 Inhibitor) as example.")
        example_smi = "C(CC1=CC(=C(C=C1F)F)F)N(CC(=O)N2CC(CN2)(C(F)(F)F)F)N"
        st.code(example_smi)
        if st.button("Load Example"):
            smiles_list = [example_smi]

    # 3. Prediction
    if st.button("🚀 Run Prediction", type="primary"):
        if not selected_ml_targets:
            st.error("Please select at least one target.")
        elif not smiles_list:
            st.error("Please provide input SMILES.")
        else:
            results = []
            
            # Load Models
            models = {}
            for t in selected_ml_targets:
                m = load_ml_model(t)
                if m: models[t] = m
                else: st.warning(f"Could not load model for {t}")
            
            if not models:
                st.error("No models loaded successfully.")
                return

            progress_bar = st.progress(0)
            
            # Process Molecules
            invalid_log = []
            for i, smi in enumerate(smiles_list):
                std_smi = standardize_smiles_rdkit(smi, invalid_log)
                if std_smi:
                    fp = calculate_ecfp4(std_smi)
                    if fp is not None:
                        # Reshape for sklearn (1, 2048)
                        fp_reshaped = fp.reshape(1, -1)
                        
                        row = {"ID": f"Mol_{i+1}", "SMILES": std_smi}
                        
                        for t, model in models.items():
                            # Predict Activity (0 or 1)
                            prediction = model.predict(fp_reshaped)[0]
                            # Predict Probability
                            proba = model.predict_proba(fp_reshaped)[0][1] # Probability of class 1
                            
                            status = "Active 🟢" if prediction == 1 else "Inactive 🔴"
                            row[f"{t} Activity"] = status
                            row[f"{t} Prob"] = f"{proba:.2f}"
                        
                        results.append(row)
                progress_bar.progress((i+1)/len(smiles_list))
                
            if results:
                st.success("Prediction Complete!")
                df_res = pd.DataFrame(results)
                st.dataframe(df_res)
                st.download_button("Download Results", convert_df_to_csv(df_res), "prediction_results.csv", "text/csv")
            else:
                st.warning("No valid molecules processed.")
            
            if invalid_log:
                st.warning(f"Skipped {len(invalid_log)} invalid SMILES.")

def display_diabetes_docking_procedure():
    st.header(f"Molecular Docking Model System Targeting Key Proteins Involved In T2DM")
    st.image("https://raw.githubusercontent.com/HenryChritopher02/GSJ/main/docking-app.png", use_column_width=True)
    
    # Initialize session state
    if 'docking_results' not in st.session_state:
        st.session_state.docking_results = []
    if 'prepared_ligand_paths' not in st.session_state:
        st.session_state.prepared_ligand_paths = []

    # --- SIDEBAR SETTINGS (Keep this logic) ---
    with st.sidebar:
        vina_ready = check_vina_binary(show_success=False)
        st.subheader("Select Targets")
        st.caption("Choose targets associated with Diabetes Type 2:")
        
        selected_targets_keys = st.multiselect(
            "Select Target(s):",
            options=list(DIABETES_TARGETS.keys()),
            default=[list(DIABETES_TARGETS.keys())[0]]
        )
        
        if st.button("Fetch Selected Targets Data", key="fetch_targets_btn"):
            if not selected_targets_keys:
                st.warning("Please select at least one target.")
            else:
                with st.spinner("Downloading receptor and config files..."):
                    download_count = 0
                    for key in selected_targets_keys:
                        info = DIABETES_TARGETS[key]
                        download_file_from_github(BASE_GITHUB_URL_FOR_DATA, f"targets/{info['pdbqt']}", info['pdbqt'], RECEPTOR_DIR_LOCAL)
                        download_file_from_github(BASE_GITHUB_URL_FOR_DATA, f"configs/{info['config']}", info['config'], CONFIG_DIR_LOCAL)
                        download_count += 1
                    st.success(f"Successfully checked/downloaded data for {download_count} targets.")

    # --- NEW TABS LAYOUT ---
    tab1, tab2, tab3 = st.tabs(["📂 1. Ligand Input", "🚀 2. Run Docking", "📊 3. Analysis & 3D"])

    # --- TAB 1: INPUT ---
    with tab1:
        st.info("Prepare ligands for docking.")
        
        # New Input Methods
        input_method = st.radio("Input Method:", ("Upload PDBQT/ZIP", "Draw Molecule", "Use Example Molecule"), horizontal=True)
        new_ligands = []

        if input_method == "Upload PDBQT/ZIP":
            uploaded_files = st.file_uploader("Select files:", type=["pdbqt", "zip"], accept_multiple_files=True)
            if st.button("Process Files") and uploaded_files:
                for up_file in uploaded_files:
                    if up_file.name.endswith(".zip"):
                         # (Existing ZIP logic shortened for brevity)
                        temp_zip = LIGAND_UPLOAD_TEMP_DIR / up_file.name
                        with open(temp_zip, "wb") as f: f.write(up_file.getbuffer())
                        with zipfile.ZipFile(temp_zip, 'r') as z: z.extractall(ZIP_EXTRACT_DIR_LOCAL)
                        for item in ZIP_EXTRACT_DIR_LOCAL.rglob("*.pdbqt"):
                            dest = LIGAND_PREP_DIR_LOCAL / item.name
                            shutil.copy(item, dest); new_ligands.append(str(dest))
                    else:
                        dest = LIGAND_PREP_DIR_LOCAL / up_file.name
                        with open(dest, "wb") as f: f.write(up_file.getbuffer())
                        new_ligands.append(str(dest))
                st.success(f"Added {len(new_ligands)} ligands.")

        elif input_method == "Draw Molecule":
            st.write("Draw a molecule and convert it to PDBQT for docking.")
            drawn_smiles = st_ketcher(key="docking_ketcher")
            lig_name_draw = st.text_input("Ligand Name:", value="drawn_ligand_01")
            
            if st.button("Convert to PDBQT") and drawn_smiles:
                # Use the provided convert_smiles_to_pdbqt function
                with st.spinner("Standardizing and converting..."):
                    std_smi = standardize_smiles_rdkit(drawn_smiles, [])
                    if std_smi:
                        result = convert_smiles_to_pdbqt(
                            std_smi, lig_name_draw, LIGAND_PREP_DIR_LOCAL, 
                            7.4, False, False, SCRUB_PY_LOCAL_PATH, MK_PREPARE_LIGAND_PY_LOCAL_PATH
                        )
                        if result:
                            new_ligands.append(result['pdbqt_path'])
                            st.success(f"Converted {lig_name_draw} successfully!")
                        else:
                            st.error("Conversion failed. Check scripts (scrub.py/mk_prepare_ligand.py).")
                    else:
                        st.error("Invalid SMILES.")

        elif input_method == "Use Example Molecule":
            st.markdown("Using **Metformin** (Antidiabetic) as example.")
            example_smi = "CN(C)C(=N)NC(=N)N"
            st.code(example_smi)
            if st.button("Process Example"):
                with st.spinner("Processing Metformin..."):
                    result = convert_smiles_to_pdbqt(
                        example_smi, "metformin_example", LIGAND_PREP_DIR_LOCAL, 
                        7.4, False, False, SCRUB_PY_LOCAL_PATH, MK_PREPARE_LIGAND_PY_LOCAL_PATH
                    )
                    if result:
                        new_ligands.append(result['pdbqt_path'])
                        st.success("Metformin added to docking list.")
                    else:
                        st.error("Conversion failed.")

        # Update Session State
        if new_ligands:
            current_paths = set(st.session_state.prepared_ligand_paths)
            for p in new_ligands: current_paths.add(p)
            st.session_state.prepared_ligand_paths = list(current_paths)

        if st.session_state.prepared_ligand_paths:
            with st.expander(f"✅ Ready Ligands ({len(st.session_state.prepared_ligand_paths)})"):
                for p in st.session_state.prepared_ligand_paths: st.text(Path(p).name)
            if st.button("Clear List"):
                st.session_state.prepared_ligand_paths = []
                st.experimental_rerun()

    # --- TAB 2: EXECUTION ---
    with tab2:
        st.write("### Simulation Controls")
        if st.button("Start Screening", type="primary"):
            if not vina_ready: st.error("Vina executable is missing.")
            elif not selected_targets_keys: st.error("No targets selected.")
            elif not st.session_state.prepared_ligand_paths: st.error("No ligands loaded.")
            else:
                targets_ready = []
                for t_key in selected_targets_keys:
                    t_info = DIABETES_TARGETS[t_key]
                    r_path = RECEPTOR_DIR_LOCAL / t_info['pdbqt']
                    c_path = CONFIG_DIR_LOCAL / t_info['config']
                    if r_path.exists() and c_path.exists(): targets_ready.append((t_key, r_path, c_path))
                    else: st.error(f"Files missing for {t_key}.")
                
                if len(targets_ready) == len(selected_targets_keys):
                    st.info(f"Docking {len(st.session_state.prepared_ligand_paths)} ligands vs {len(targets_ready)} targets.")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_tasks = len(st.session_state.prepared_ligand_paths) * len(targets_ready)
                    completed_tasks = 0
                    results_data = []
                    DOCKING_OUTPUT_DIR_LOCAL.mkdir(parents=True, exist_ok=True)

                    for lig_path_str in st.session_state.prepared_ligand_paths:
                        lig_path = Path(lig_path_str)
                        lig_name = lig_path.stem
                        row_data = {"Ligand": lig_name}
                        
                        for t_name, r_path, c_path in targets_ready:
                            status_text.text(f"Docking {lig_name} against {t_name}...")
                            out_filename = f"{lig_name}_{DIABETES_TARGETS[t_name]['pdbqt'].replace('.pdbqt', '')}_out.pdbqt"
                            out_path = DOCKING_OUTPUT_DIR_LOCAL / out_filename
                            
                            ret_code, stdout, stderr = run_single_docking(VINA_PATH_LOCAL, r_path, lig_path, c_path, out_path)
                            
                            if ret_code == 0 and out_path.exists():
                                score = parse_vina_score_from_file(out_path)
                                row_data[t_name] = score if score is not None else "N/A"
                            else: row_data[t_name] = "Error"
                            completed_tasks += 1
                            progress_bar.progress(completed_tasks / total_tasks)
                        results_data.append(row_data)

                    st.session_state.docking_results = results_data
                    status_text.text("Docking completed!")
                    st.success("Run Finished.")
                    st.balloons()

    # --- TAB 3: ANALYSIS ---
    with tab3:
        if st.session_state.docking_results:
            df_results = pd.DataFrame(st.session_state.docking_results)
            score_cols = [col for col in df_results.columns if col != 'Ligand']
            for col in score_cols: df_results[col] = pd.to_numeric(df_results[col], errors='coerce')

            # 1. HEATMAP TABLE
            st.subheader("🔥 Affinity Heatmap")
            st.dataframe(
                df_results.style.background_gradient(
                    cmap='RdYlGn_r', subset=score_cols, vmin=-12, vmax=-4
                ).format(precision=2, na_rep="N/A"),
                use_container_width=True
            )
            
            col_dl, col_chart = st.columns([1, 2])
            with col_dl:
                csv = convert_df_to_csv(df_results)
                st.download_button("Download CSV", csv, "docking_results.csv", "text/csv")

            # 2. DISTRIBUTION CHART
            st.markdown("---")
            st.subheader("📈 Score Distribution")
            try:
                df_melted = df_results.melt(id_vars=['Ligand'], var_name='Target', value_name='Score')
                df_melted = df_melted.dropna()
                fig = px.box(df_melted, x='Target', y='Score', points="all", color='Target', title="Binding Energy Distribution")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Not enough data for chart.")

            # 3. 3D VISUALIZATION
            st.markdown("---")
            st.subheader("🧬 3D Complex Visualization")
            
            c1, c2 = st.columns(2)
            with c1:
                selected_ligand = st.selectbox("Select Ligand:", df_results['Ligand'].unique())
            with c2:
                selected_target = st.selectbox("Select Target:", score_cols)

            if st.button("Render 3D Structure"):
                target_info = DIABETES_TARGETS[selected_target]
                receptor_file = RECEPTOR_DIR_LOCAL / target_info['pdbqt']
                out_filename = f"{selected_ligand}_{target_info['pdbqt'].replace('.pdbqt', '')}_out.pdbqt"
                docked_ligand_file = DOCKING_OUTPUT_DIR_LOCAL / out_filename

                if receptor_file.exists() and docked_ligand_file.exists():
                    # Output path for PDB
                    pdb_viz_file = docked_ligand_file.with_suffix(".pdb")
                    
                    with st.spinner("Extracting best pose & converting to PDB..."):
                        # Use the new PDB conversion function
                        convert_success = convert_pdbqt_to_pdb(docked_ligand_file, pdb_viz_file)
                    
                    if convert_success:
                        st.write(f"Visualizing: **{selected_ligand}** (Best Pose) bound to **{selected_target}**")
                        # Pass the new PDB file to the viewer
                        view_complex(str(receptor_file), str(pdb_viz_file))
                    else:
                        st.error("Visualization preparation failed.")
                else:
                    st.error(f"Output file not found: {out_filename}. Did the docking finish successfully?")
        else:
            st.info("No docking results to analyze yet. Please run docking in Tab 2.")

def display_about_page():
    st.header("About T2DM Docking App")
    st.markdown(f"**Molecular Docking Model System Targeting Key Proteins Involved In T2DM**")
    st.markdown("""
    This application is specialized for screening compounds against key therapeutic targets for T2DM.
    
    **Features:**
    - **Focused Targets:** Pre-configured screening against 5 major diabetes-related proteins:
        1. **DPP-4 (4A5S):** Enzyme that degrades incretin hormones.
        2. **GLP1-R (6X19):** Receptor that stimulates glucose-dependent insulin secretion.
        3. **PPAR-γ (5Y2O):** Nuclear receptor regulating fatty acid storage and glucose metabolism.
        4. **SGLT2 (8HEZ):** Transporter responsible for glucose reabsorption in the kidneys.
        5. **SUR1 (7S5V):** Regulatory subunit of K-ATP channels controlling insulin release.
    - **Simplified Input:** Direct upload of `.pdbqt` files or `.zip` archives.
    - **Automated Vina:** Runs AutoDock Vina automatically for all combinations.
    """)

def main():
    st.set_page_config(layout="wide", page_title=f"Diabetes Docking v{APP_VERSION}")
    
    initialize_directories()

    #st.sidebar.image("https://raw.githubusercontent.com/HenryChritopher02/GSJ/main/docking-app.png", width=300)
    st.sidebar.title("Navigation")

    app_mode = st.sidebar.radio("Go to:", ("T2DM Docking", "T2DM AI prediction", "About"))
    st.sidebar.markdown("---")

    if app_mode == "T2DM Docking":
        display_diabetes_docking_procedure()
    elif app_mode == "T2DM AI prediction":
        display_ml_prediction_procedure()
    elif app_mode == "About":
        display_about_page()

if __name__ == "__main__":
    main()
