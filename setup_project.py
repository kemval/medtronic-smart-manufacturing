"""
Smart Manufacturing Dashboard for Medtronic
Project Structure Creator

INSTRUCTIONS:
1. Save this entire file as: setup_project.py
2. Run with: python3 setup_project.py
3. This will create all folders and files needed for the project
"""

import os

def create_project_structure():
    """Create all necessary folders and files"""
    
    # Define project structure
    structure = {
        'data': {
            'raw': ['README.md'],
            'processed': ['README.md'],
            'models': ['README.md']
        },
        'src': {
            'data_generation': ['__init__.py'],
            'data_processing': ['__init__.py'],
            'models': ['__init__.py'],
            'visualization': ['__init__.py']
        },
        'dashboard': ['__init__.py'],
        'notebooks': ['README.md'],
        'outputs': {
            'reports': ['README.md'],
            'figures': ['README.md']
        }
    }
    
    # Create root files
    root_files = {
        '.gitignore': '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# Data
*.csv
*.db
*.sqlite
*.xlsx
data/raw/*
!data/raw/README.md
data/processed/*
!data/processed/README.md

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml
''',
        'README.md': '''# Smart Manufacturing Dashboard for Medtronic

## 🎯 Project Overview
An AI-powered manufacturing monitoring system demonstrating predictive maintenance, 
quality control, and real-time production analytics for medical device manufacturing.

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard/app.py
```

## 📊 Features
- Real-time production monitoring
- Predictive maintenance alerts
- Quality control anomaly detection
- Interactive Power BI-style dashboards
- Automated reporting

## 🛠️ Tech Stack
- Python 3.12
- Streamlit for dashboard
- Scikit-learn for ML models
- Plotly for visualizations
- SQLite for data storage

## 📁 Project Structure
```
├── data/              # Data storage
├── src/               # Source code
├── dashboard/         # Streamlit app
├── notebooks/         # Jupyter notebooks
└── outputs/           # Reports and figures
```

## 👨‍💻 Author
Built as a demo project for Medtronic Engineering Intern application
'''
    }
    
    print("🏗️  Creating Smart Manufacturing Dashboard Project Structure...\n")
    
    # Create directories
    def create_structure(base_path, struct):
        for folder, contents in struct.items():
            folder_path = os.path.join(base_path, folder)
            os.makedirs(folder_path, exist_ok=True)
            print(f"✅ Created: {folder_path}")
            
            if isinstance(contents, dict):
                create_structure(folder_path, contents)
            elif isinstance(contents, list):
                for file in contents:
                    file_path = os.path.join(folder_path, file)
                    if not os.path.exists(file_path):
                        with open(file_path, 'w') as f:
                            if file == 'README.md':
                                f.write(f"# {folder.title()}\n\nThis directory contains {folder} files.")
                            elif file == '__init__.py':
                                f.write(f"# {folder} module\n")
                        print(f"  📄 Created: {file_path}")
    
    # Create structure
    create_structure('.', structure)
    
    # Create root files
    print("\n📝 Creating root files...")
    for filename, content in root_files.items():
        with open(filename, 'w') as f:
            f.write(content)
        print(f"✅ Created: {filename}")
    
    print("\n" + "="*60)
    print("🎉 PROJECT STRUCTURE CREATED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download CNC dataset from Kaggle (I'll help with this)")
    print("3. Run data generation scripts")
    print("4. Launch dashboard: streamlit run dashboard/app.py")
    print("\n💡 Tip: Open this project in Windsurf for the best experience!")

if __name__ == "__main__":
    create_project_structure()