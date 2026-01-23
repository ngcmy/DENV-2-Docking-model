import re
import streamlit as st
import subprocess
import os
import zipfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from streamlit_ketcher import st_ketcher
import plotly.express as px
import py3Dmol
from stmol import showmol
from meeko import MoleculePreparation, PDBQTMolecule

# --- IMPORTS TỪ UTILS ---
from utils.paths import (
    BASE_GITHUB_URL_FOR_DATA, 
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

# --- CẤU HÌNH CÁC MỤC TIÊU (DATA STRUCTURE) ---
# Lưu ý: Để hỗ trợ 2 khoang A/B, bạn có thể thêm key "config_B" vào dict này nếu có file thực tế.
# Hiện tại code sẽ sử dụng "config" mặc định như khoang A.
DENV2_TARGETS = {
    "NS1 (4O6B)": {
        "pdbqt": "NS1.pdbqt",
        "config": "NS1.txt" 
    },
    "NS3 Helicase (2BHR)": {
        "pdbqt": "NS3 Helicase.pdbqt",
        "config": "NS3 Helicase.txt"
    },
    "NS2B-NS3 (6MO1)": {
        "pdbqt": "NS2B-NS3.pdbqt",
        "config": "NS2B-NS3.txt"
    },
    "NS5 MTase (2P3O)": {
        "pdbqt": "NS5 MTase.pdbqt",
        "config": "NS5 MTase.txt"
    },
    "NS5 RdRp (7HKD)": {
        "pdbqt": "NS5 RdRp.pdbqt",
        "config": "NS5 RdRp.txt"
    }
}

# --- CÁC HÀM HỖ TRỢ XỬ LÝ FILE ---

def convert_pdbqt_to_pdb(pdbqt_path, output_pdb_path):
    """Chuyển đổi PDBQT sang PDB để hiển thị đẹp hơn (giữ nguyên logic cũ)"""
    try:
        with open(pdbqt_path, 'r') as f:
            lines = f.readlines()
        pdb_lines = []
        in_model = False
        model_found = False
        for line in lines:
            if line.startswith("MODEL"):
                in_model = True; model_found = True; continue
            if line.startswith("ENDMDL"): break 
            if line.startswith("ATOM") or line.startswith("HETATM"):
                clean_line = line[:66] + "\n"
                pdb_lines.append(clean_line)
            elif not model_found and not line.startswith("TORSDOF"): 
                pdb_lines.append(line)
        with open(output_pdb_path, 'w') as f: f.writelines(pdb_lines)
        return True
    except Exception as e:
        print(f"PDB Conversion Error: {e}")
        return False

def parse_vina_config(config_path):
    """
    Đọc file config của Vina để lấy tọa độ và kích thước hộp (Grid Box).
    Trả về dict: {'center': {'x':...}, 'size': {'x':...}}
    """
    params = {'center_x': 0, 'center_y': 0, 'center_z': 0, 'size_x': 0, 'size_y': 0, 'size_z': 0}
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split('=')
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = float(parts[1].strip())
                    if key in params:
                        params[key] = val
        return {
            'center': {'x': params['center_x'], 'y': params['center_y'], 'z': params['center_z']},
            'dimensions': {'w': params['size_x'], 'h': params['size_y'], 'd': params['size_z']}
        }
    except Exception as e:
        st.error(f"Error parsing config: {e}")
        return None

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
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def view_complex_with_box(protein_pdb_content, box_config=None, box_color='green', ligand_content=None, ligand_format='pdbqt'):
    """
    Hiển thị Protein, Ligand (nếu có) và vẽ hộp Grid Box dựa trên Config.
    """
    view = py3Dmol.view(width=800, height=500)
    
    # Add Protein
    view.addModelsAsFrames(protein_pdb_content)
    view.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}})
    
    # Add Ligand if exists
    if ligand_content:
        view.addModel(ligand_content, ligand_format)
        view.setStyle({'model': -1}, {"stick": {'colorscheme': 'greenCarbon'}})

    # Draw Grid Box (Khoanh vùng)
    if box_config:
        view.addBox({
            'center': box_config['center'],
            'dimensions': box_config['dimensions'],
            'color': box_color,
            'opacity': 0.5,
            'wireframe': True
        })
        # Thêm nhãn cho hộp
        view.addLabel("Binding Site", 
                      {'position': box_config['center'], 'backgroundColor': 'black', 'fontColor': 'white'})

    view.zoomTo()
    showmol(view, height=500, width=800)

# --- MAIN APP ---

def display_denv2_docking_procedure():
    st.header(f"Docking Model System Targeting In DENV-2 Key Proteins")
    st.image("https://github.com/ngcmy/DENV-2-Docking-system/blob/main/App.png?raw=true", use_column_width=True)
    
    # Initialize session state
    if 'docking_results' not in st.session_state: st.session_state.docking_results = []
    if 'prepared_ligand_paths' not in st.session_state: st.session_state.prepared_ligand_paths = []
    if 'selected_targets_global' not in st.session_state: st.session_state.selected_targets_global = []

    # Check Vina
    vina_ready = check_vina_binary(show_success=False)

    # --- TABS LAYOUT ---
    tab1, tab2, tab3, tab4 = st.tabs(["📂 1. Ligand Input", "🎯 2. Select Target & Viz", "🚀 3. Run Docking", "📊 4. Analysis & 3D"])

    # ==========================================
    # PHẦN 1: LIGAND INPUT 
    # ==========================================
    with tab1:
        st.info("Prepare ligands for docking.")
        input_method = st.radio("Input Method:", ("Upload PDBQT/ZIP", "Draw Molecule", "Use Example Molecule"), horizontal=True)
        new_ligands = []

        if input_method == "Upload PDBQT/ZIP":
            uploaded_files = st.file_uploader("Select files:", type=["pdbqt", "zip"], accept_multiple_files=True)
            if st.button("Process Files") and uploaded_files:
                for up_file in uploaded_files:
                    if up_file.name.endswith(".zip"):
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
            drawn_smiles = st_ketcher(key="docking_ketcher")
            lig_name_draw = st.text_input("Ligand Name:", value="drawn_ligand")
            if st.button("Convert to PDBQT") and drawn_smiles:
                with st.spinner("Converting..."):
                    std_smi = standardize_smiles_rdkit(drawn_smiles, [])
                    if std_smi:
                        result = convert_smiles_to_pdbqt(std_smi, lig_name_draw, LIGAND_PREP_DIR_LOCAL, 7.4, False, False, SCRUB_PY_LOCAL_PATH, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                        if result: new_ligands.append(result['pdbqt_path']); st.success("Converted!")
                    else: st.error("Invalid SMILES.")

        elif input_method == "Use Example Molecule":
            st.markdown("Using **Mosnodenvir (JNJ-1802)**")
            if st.button("Process Example"):
                result = convert_smiles_to_pdbqt("COC1=CC(N[C@H](C(=O)C2=CNC3=CC=C(OC(F)(F)F)C=C23)C2=CC=C(Cl)C=C2OC)=CC(=C1)S(C)(=O)=O", "mosnodenvir", LIGAND_PREP_DIR_LOCAL, 7.4, False, False, SCRUB_PY_LOCAL_PATH, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                if result: new_ligands.append(result['pdbqt_path']); st.success("Added Example.")

        if new_ligands:
            current_paths = set(st.session_state.prepared_ligand_paths)
            for p in new_ligands: current_paths.add(p)
            st.session_state.prepared_ligand_paths = list(current_paths)
        
        if st.session_state.prepared_ligand_paths:
            st.write(f"✅ **{len(st.session_state.prepared_ligand_paths)} Ligands ready.**")

    # ==========================================
    # PHẦN 2: SELECT TARGET & VISUALIZATION
    # ==========================================
    with tab2:
        st.subheader("Select Targets for Screening")
        
        # 1. Chọn Targets để chạy Docking sau này
        selected_targets = st.multiselect(
            "Choose Target(s) to include in docking run:",
            options=list(DENV2_TARGETS.keys()),
            default=[list(DENV2_TARGETS.keys())[0]]
        )
        # Lưu vào session state để Tab 3 dùng
        st.session_state.selected_targets_global = selected_targets

        if st.button("Fetch/Update Selected Targets Data"):
             with st.spinner("Checking and downloading files..."):
                cnt = 0
                for key in selected_targets:
                    info = DENV2_TARGETS[key]
                    download_file_from_github(BASE_GITHUB_URL_FOR_DATA, f"Target/{info['pdbqt']}", info['pdbqt'], RECEPTOR_DIR_LOCAL)
                    download_file_from_github(BASE_GITHUB_URL_FOR_DATA, f"Config/{info['config']}", info['config'], CONFIG_DIR_LOCAL)
                    cnt += 1
                st.success(f"Ready: {cnt} targets.")

        st.markdown("---")
        st.subheader("🔍 Target Visualization & Grid Box Inspection")
        st.markdown("Visualize the target structure and the defined docking box (Khoang A/B).")

        col_vis_sel, col_vis_opt = st.columns([1, 1])
        with col_vis_sel:
            viz_target_key = st.selectbox("Select Target to Visualize:", options=list(DENV2_TARGETS.keys()))
        
        with col_vis_opt:
            # Giả lập lựa chọn Config A hoặc B. 
            # Vì trong biến DENV2_TARGETS chỉ có 1 config, ta sẽ mặc định dùng nó.
            # Nếu bạn có file config B thực tế, bạn có thể cập nhật logic ở đây để load file khác.
            box_choice = st.radio("Select Binding Pocket Configuration:", ["Config A (Default)", "Config B (Custom/Alternative)"], horizontal=True)

        if viz_target_key:
            info = DENV2_TARGETS[viz_target_key]
            pdbqt_path = RECEPTOR_DIR_LOCAL / info['pdbqt']
            
            # Xử lý đường dẫn file config dựa trên lựa chọn A hoặc B
            if box_choice == "Config A (Default)":
                config_path = CONFIG_DIR_LOCAL / info['config']
                box_color = "green"
            else:
                # Logic giả định cho Config B: Ví dụ file tên là "NS1_B.txt"
                # Hiện tại fallback về config A nhưng đổi màu để demo
                config_path = CONFIG_DIR_LOCAL / info['config'] 
                box_color = "red"
                st.caption("Using default config for demo (replace code logic to point to actual Config B file).")

            # Nút render
            if st.button("Render 3D Target & Box"):
                if pdbqt_path.exists() and config_path.exists():
                    # 1. Parse Box
                    box_data = parse_vina_config(config_path)
                    
                    # 2. Convert PDBQT to PDB string for viewing
                    pdb_viz_path = pdbqt_path.with_suffix(".pdb")
                    convert_pdbqt_to_pdb(pdbqt_path, pdb_viz_path)
                    
                    if pdb_viz_path.exists():
                        with open(pdb_viz_path, 'r') as f: pdb_content = f.read()
                        
                        st.markdown(f"**Visualizing:** {viz_target_key} | **Box:** {box_choice}")
                        # 3. Call visualizer
                        view_complex_with_box(pdb_content, box_config=box_data, box_color=box_color)
                        
                        # Show box details
                        if box_data:
                            with st.expander("Grid Box Coordinates details"):
                                st.write(box_data)
                    else:
                        st.error("Conversion to PDB failed.")
                else:
                    st.warning("Files not found. Please click 'Fetch/Update Selected Targets Data' above first.")

    # ==========================================
    # PHẦN 3: RUN DOCKING
    # ==========================================
    with tab3:
        st.write("### Simulation Controls")
        # Lấy danh sách targets từ Tab 2
        active_targets = st.session_state.selected_targets_global
        
        if st.button("Start Screening", type="primary"):
            if not vina_ready: st.error("Vina executable missing.")
            elif not active_targets: st.error("Please select targets in Tab 2.")
            elif not st.session_state.prepared_ligand_paths: st.error("No ligands loaded in Tab 1.")
            else:
                # Chuẩn bị file
                targets_ready = []
                for t_key in active_targets:
                    t_info = DENV2_TARGETS[t_key]
                    r_path = RECEPTOR_DIR_LOCAL / t_info['pdbqt']
                    c_path = CONFIG_DIR_LOCAL / t_info['config']
                    if r_path.exists() and c_path.exists(): targets_ready.append((t_key, r_path, c_path))
                
                if not targets_ready:
                    st.error("Target files missing. Go to Tab 2 and Fetch Data.")
                else:
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
                            out_filename = f"{lig_name}_{DENV2_TARGETS[t_name]['pdbqt'].replace('.pdbqt', '')}_out.pdbqt"
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

    # ==========================================
    # PHẦN 4: ANALYSIS & 3D
    # ==========================================
    with tab4:
        if st.session_state.docking_results:
            df_results = pd.DataFrame(st.session_state.docking_results)
            score_cols = [col for col in df_results.columns if col != 'Ligand']
            for col in score_cols: df_results[col] = pd.to_numeric(df_results[col], errors='coerce')

            st.subheader("🔥 Affinity Heatmap")
            st.dataframe(df_results.style.background_gradient(cmap='RdYlGn_r', subset=score_cols, vmin=-12, vmax=-4).format(precision=2, na_rep="N/A"), use_container_width=True)
            st.download_button("Download CSV", convert_df_to_csv(df_results), "docking_results.csv", "text/csv")

            st.markdown("---")
            st.subheader("🧬 3D Complex Visualization (Result)")
            
            c1, c2 = st.columns(2)
            with c1: selected_ligand = st.selectbox("Select Ligand:", df_results['Ligand'].unique())
            with c2: selected_target = st.selectbox("Select Target Result:", score_cols)

            if st.button("View Docked Complex"):
                target_info = DENV2_TARGETS[selected_target]
                receptor_file = RECEPTOR_DIR_LOCAL / target_info['pdbqt']
                out_filename = f"{selected_ligand}_{target_info['pdbqt'].replace('.pdbqt', '')}_out.pdbqt"
                docked_ligand_file = DOCKING_OUTPUT_DIR_LOCAL / out_filename

                if receptor_file.exists() and docked_ligand_file.exists():
                    pdb_viz_file = docked_ligand_file.with_suffix(".pdb")
                    convert_pdbqt_to_pdb(docked_ligand_file, pdb_viz_file)
                    convert_pdbqt_to_pdb(receptor_file, receptor_file.with_suffix(".pdb")) # Ensure receptor is PDB too

                    with open(receptor_file.with_suffix(".pdb"), 'r') as f: r_data = f.read()
                    with open(pdb_viz_file, 'r') as f: l_data = f.read()

                    st.write(f"**{selected_ligand}** bound to **{selected_target}**")
                    # Dùng hàm view mới, nhưng không cần vẽ box khi xem kết quả (tuỳ chọn)
                    view_complex_with_box(r_data, ligand_content=l_data, ligand_format='pdb')
                else:
                    st.error("Output file not found.")
        else:
            st.info("No docking results yet. Run Tab 3 first.")

def display_about_page():
    st.header("About DENV-2 Docking App")
    st.markdown(f"**Docking Model System Targeting In DENV-2 Key Proteins**")
    st.markdown("""
    This application is specialized for screening compounds against key therapeutic targets for DENV2.
    
    **Features:**
    - **Focused Targets:** Pre-configured screening against these major DENV2 key proteins:
        1. **NS1:** A multifunctional glycoprotein essential for viral replication and a primary mediator of vascular leakage in the host.
        2. **NS2-NS3B Protease:** A critical enzyme complex responsible for the proteolytic cleavage of the viral polyprotein into functional units.
        3. **NS3 Helicase:** An enzyme that unwinds double-stranded RNA templates to facilitate the viral genome replication process.
        4. **NS5 MTase:** Responsible for the 5' capping of viral RNA to ensure its stability and enable evasion of the host immune system.
        5. **NS5 RdRp:** The core polymerase enzyme that directly catalyzes the synthesis and elongation of the viral RNA genome.
    - **Simplified Input:** Direct upload of `.pdbqt` files or `.zip` archives.
    - **Automated Vina:** Runs AutoDock Vina automatically for all combinations.
    """)

def main():
    st.set_page_config(layout="wide", page_title="DENV-2 Docking System")
    initialize_directories()
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to:", ("DENV-2 Docking", "About"))
    
    if app_mode == "DENV-2 Docking":
        display_denv2_docking_procedure()
    elif app_mode == "About":
        display_about_page()

if __name__ == "__main__":
    main()
