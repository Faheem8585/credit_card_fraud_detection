"""
Generate pipeline visualization diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, 'Credit Card Fraud Detection Pipeline', 
        fontsize=20, fontweight='bold', ha='center')

# Colors
color_data = '#3498db'
color_preprocess = '#2ecc71'
color_model = '#e74c3c'
color_api = '#f39c12'
color_frontend = '#9b59b6'

def create_box(ax, x, y, width, height, text, color, textcolor='white'):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         facecolor=color,
                         edgecolor='black',
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
           fontsize=11, ha='center', va='center',
           color=textcolor, fontweight='bold')

def create_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->,head_width=0.4,head_length=0.4',
                           color='black', linewidth=2,
                           mutation_scale=20)
    ax.add_patch(arrow)

# Stage 1: Data
create_box(ax, 1, 9.5, 3, 0.8, 'Kaggle Dataset\n284,807 transactions', color_data)

# Stage 2: Preprocessing
create_arrow(ax, 2.5, 9.5, 2.5, 8.8)
create_box(ax, 1, 7.5, 3, 1.2, 'Data Preprocessing\n• StandardScaler\n• Train/Val/Test Split\n• SMOTE (class balance)', color_preprocess)

# Stage 3: Model Training (4 parallel models)
create_arrow(ax, 2.5, 7.5, 1, 6.5)
create_arrow(ax, 2.5, 7.5, 3, 6.5)
create_arrow(ax, 2.5, 7.5, 5, 6.5)
create_arrow(ax, 2.5, 7.5, 7, 6.5)

create_box(ax, 0.2, 5.5, 1.6, 0.9, 'Logistic\nRegression\n0.9660 AUC', color_model, 'white')
create_box(ax, 2.2, 5.5, 1.6, 0.9, 'Random\nForest\n0.9621 AUC', color_model, 'white')
create_box(ax, 4.2, 5.5, 1.6, 0.9, 'Gradient\nBoosting\n0.9766 AUC', color_model, 'white')
create_box(ax, 6.2, 5.5, 1.6, 0.9, 'Isolation\nForest\n0.8254 AUC', color_model, 'white')

# Stage 4: Ensemble
create_arrow(ax, 1, 5.5, 4, 4.8)
create_arrow(ax, 3, 5.5, 4, 4.8)
create_arrow(ax, 5, 5.5, 5, 4.8)
create_arrow(ax, 7, 5.5, 6, 4.8)

create_box(ax, 3.5, 4, 3, 0.7, 'Weighted Ensemble\n0.9668 AUC', '#c0392b', 'white')

# Stage 5: Model Persistence
create_arrow(ax, 5, 4, 5, 3.5)
create_box(ax, 3.5, 2.7, 3, 0.7, 'Saved Models\n(.pkl files)', color_preprocess)

# Stage 6: API Layer
create_arrow(ax, 5, 2.7, 8.5, 2.2)
create_box(ax, 7, 1.5, 2.5, 0.7, 'FastAPI Backend\nJWT Auth + Database', color_api, 'white')

# Stage 7: Frontend
create_arrow(ax, 8.25, 1.5, 8.25, 0.8)
create_box(ax, 7, 0.1, 2.5, 0.7, 'Streamlit Dashboard\nUser Interface', color_frontend, 'white')

# Add deployment path
create_box(ax, 0.2, 2, 2.5, 1.2, 'Database Layer\n• SQLite/PostgreSQL\n• Users\n• Transactions\n• Predictions', '#34495e', 'white')
create_arrow(ax, 2.7, 2.6, 7, 1.9)

# Add evaluation metrics box
create_box(ax, 0.2, 0.1, 2.5, 1.2, 'Visualizations\n• ROC Curves\n• Confusion Matrix\n• Precision-Recall\n• Metrics Table', '#16a085', 'white')
create_arrow(ax, 2.7, 0.7, 7, 0.5)

# Legend
legend_elements = [
    mpatches.Patch(color=color_data, label='Data'),
    mpatches.Patch(color=color_preprocess, label='Preprocessing'),
    mpatches.Patch(color=color_model, label='ML Models'),
    mpatches.Patch(color=color_api, label='Backend'),
    mpatches.Patch(color=color_frontend, label='Frontend')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('dashboard/assets/pipeline_diagram.png', dpi=200, bbox_inches='tight', facecolor='white')
print("✅ Pipeline diagram saved to dashboard/assets/pipeline_diagram.png")
