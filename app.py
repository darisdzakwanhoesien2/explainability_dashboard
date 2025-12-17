import streamlit as st
import os
import nbformat
from nbconvert import PythonExporter
from datetime import datetime

# =====================================================
# Configuration
# =====================================================

BASE_DIR = "projects"
os.makedirs(BASE_DIR, exist_ok=True)

st.set_page_config(layout="wide")
st.title("📓➡️🐍 Jupyter (.ipynb) → Python (.py) Converter")

st.markdown("""
Upload a **Jupyter Notebook** and this app will:

• Create a dedicated project folder  
• Save the `.ipynb`  
• Convert it to `.py`  
• Let you download the Python file  
""")

# =====================================================
# Utilities
# =====================================================

def get_next_project_name():
    existing = [
        d for d in os.listdir(BASE_DIR)
        if d.startswith("project_") and os.path.isdir(os.path.join(BASE_DIR, d))
    ]
    if not existing:
        return "project_1"
    nums = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    return f"project_{max(nums) + 1}"


def convert_ipynb_to_py(ipynb_path, py_path):
    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(notebook)

    with open(py_path, "w", encoding="utf-8") as f:
        f.write(source)


# =====================================================
# Upload UI
# =====================================================

uploaded_file = st.file_uploader(
    "Upload a .ipynb file",
    type=["ipynb"]
)

if uploaded_file:
    project_name = get_next_project_name()
    project_dir = os.path.join(BASE_DIR, project_name)
    os.makedirs(project_dir, exist_ok=True)

    ipynb_path = os.path.join(project_dir, f"{project_name}.ipynb")
    py_path = os.path.join(project_dir, f"{project_name}.py")

    # Save notebook
    with open(ipynb_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert
    try:
        convert_ipynb_to_py(ipynb_path, py_path)
        st.success(f"✅ Converted successfully → `{project_name}.py`")

        # Preview
        with st.expander("📄 Preview generated Python code"):
            with open(py_path, "r", encoding="utf-8") as f:
                st.code(f.read(), language="python")

        # Download
        with open(py_path, "rb") as f:
            st.download_button(
                label="⬇️ Download .py file",
                data=f,
                file_name=f"{project_name}.py",
                mime="text/x-python"
            )

    except Exception as e:
        st.error("❌ Conversion failed")
        st.exception(e)

# =====================================================
# Project Browser
# =====================================================

st.markdown("---")
st.subheader("📁 Existing Projects")

projects = sorted(os.listdir(BASE_DIR))

if projects:
    for p in projects:
        st.write(f"• `{p}/`")
else:
    st.info("No projects yet.")
