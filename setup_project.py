
from pathlib import Path

# Root directory: the folder where this script is located
ROOT = Path(__file__).resolve().parent

# All folders required for the major project
folders = [
    "config",
    "data/raw",
    "data/external",
    "data/interim",
    "data/processed",
    "notebooks",
    "src/pufferfish/data",
    "src/pufferfish/geo",
    "src/pufferfish/environmental",
    "src/pufferfish/features",
    "src/pufferfish/models",
    "src/pufferfish/twin",
    "artifacts/models",
    "artifacts/metrics",
    "app/pages",
    "app/components",
    "tests",
    "docs",
    "reports/figures",
    "reports/tables",
]

# Create the folders
for folder in folders:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)

# Files to create, with starter content where useful
files = {
    "README.md": """# Pufferfish Prediction and Digital Twin

A research-oriented B.Tech major project for predicting
pufferfish occurrence in Kerala's ring seine fishing grounds
and exploring predictions through an interactive geospatial
digital twin.

## Project status
Currently in the data audit and research phase.
""",
    "requirements.txt": """pandas
numpy
openpyxl
matplotlib
seaborn
scikit-learn
jupyter
geopandas
shapely
pyproj
folium
plotly
streamlit
joblib
pyyaml
pytest
""",
    ".gitignore": """# Python
__pycache__/
*.py[cod]
.venv/
venv/
.ipynb_checkpoints/

# Local data: do not commit private or restricted datasets
data/raw/*
data/external/*Aquasentinel\.venv310\Scripts\python.exe
data/interim/*
data/processed/*
!data/raw/.gitkeep
!data/external/.gitkeep
!data/interim/.gitkeep
!data/processed/.gitkeep

# Model artifacts and local outputs
artifacts/models/*
artifacts/metrics/*
reports/figures/*
reports/tables/*
!artifacts/models/.gitkeep
!artifacts/metrics/.gitkeep
!reports/figures/.gitkeep
!reports/tables/.gitkeep

# Secrets and local configuration
.env
""",
    ".env.example": "# Add names of required environment variables here.\n",
    "config/config.yaml": """project:
  name: pufferfish-digital-twin
  region: Kerala
  target_column: "PUFFER Presence (1/0)"

data:
  raw_dir: data/raw
  external_dir: data/external
  processed_dir: data/processed
""",
    "config/features.yaml": """# Final features will be defined after data auditing.
# Only use variables available at prediction time.
features: []
target: "PUFFER Presence (1/0)"
""",
    "src/pufferfish/__init__.py": "",
    "src/pufferfish/data/__init__.py": "",
    "src/pufferfish/geo/__init__.py": "",
    "src/pufferfish/environmental/__init__.py": "",
    "src/pufferfish/features/__init__.py": "",
    "src/pufferfish/models/__init__.py": "",
    "src/pufferfish/twin/__init__.py": "",
    "app/main.py": """import streamlit as st

st.set_page_config(
    page_title="Pufferfish Digital Twin",
    layout="wide",
)

st.title("Pufferfish Prediction and Digital Twin")
st.write("Project interface — implementation in progress.")
""",
    "app/components/maps.py": "# Reusable geospatial map components will go here.\n",
    "app/components/charts.py": "# Reusable chart components will go here.\n",
    "docs/problem_statement.md": "# Problem Statement\n\nTo be defined from the literature review and data audit.\n",
    "docs/literature_review.md": "# Literature Review\n",
    "docs/research_gap.md": "# Research Gap\n",
    "docs/existing_vs_proposed.md": "# Existing vs Proposed System\n",
    "docs/methodology.md": "# Research Methodology\n",
    "docs/system_architecture.md": "# System Architecture\n",
    "docs/data_dictionary.md": "# Data Dictionary\n",
    "docs/experiment_protocol.md": "# Experiment Protocol\n",
    "docs/references.md": "# References\n",
}

# Create starter files without overwriting existing work
for filename, content in files.items():
    path = ROOT / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.write_text(content, encoding="utf-8")

# Keep empty folders visible to Git
gitkeep_folders = [
    "data/raw",
    "data/external",
    "data/interim",
    "data/processed",
    "artifacts/models",
    "artifacts/metrics",
    "reports/figures",
    "reports/tables",
]

for folder in gitkeep_folders:
    gitkeep = ROOT / folder / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

print("Project structure created successfully!")
print(f"Project location: {ROOT}")
print("Existing files were preserved.")
