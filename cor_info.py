import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QSlider, QLabel, 
                             QTextEdit, QTableWidget, QTableWidgetItem, 
                             QSplitter, QGroupBox, QMessageBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt
from PyQt5 import uic

# --- Configuration -----------------------------------------------------------
ANTOINE_FILE = 'antoine_parameters_filtered'
CORROSION_FILE = 'corrosion_matrix'
SEARCH_DIRS = [os.getcwd()]
DEFAULT_TEMP_K = 298.15
ANTOINE_PARAM_COUNT = 5

# --- File / IO helpers -------------------------------------------------------
def find_file(base_name):
    """Return the first existing path for base_name with .xlsx or .csv."""
    for directory in SEARCH_DIRS:
        for ext in ('.xlsx', '.csv'):
            path = os.path.join(directory, base_name + ext)
            if os.path.exists(path):
                return path
    return None

def read_columns(path):
    """Read only the header row and return column names as a list."""
    reader = pd.read_csv if path.endswith('.csv') else pd.read_excel
    return reader(path, nrows=0).columns.tolist()

def read_matching(path, columns, pattern):
    """Load only the index column + columns matching `pattern` (case-insensitive)."""
    matches = [c for c in columns if re.match(pattern, str(c), re.IGNORECASE)]
    if not matches:
        return None
    if path.endswith('.csv'):
        return pd.read_csv(path, usecols=[columns[0]] + matches, index_col=0)
    return pd.read_excel(path, index_col=0)[matches]

# --- Name normalisation ------------------------------------------------------
def clean_name(name):
    """Strip a trailing '(nist)...' suffix and pandas '.1/.2' dedupe suffixes."""
    name = str(name)
    if '(nist)' in name.lower():
        name = re.split(r'\(nist\)', name, flags=re.IGNORECASE)[0]
    name = re.sub(r'(\.\d+)$', '', name)
    return name.strip()

def unique_compounds(columns):
    return {clean_name(c) for c in columns if not str(c).startswith('Unnamed')}

def compound_pattern(base_name):
    """Match the base name plus any '(nist)...' or '.N' variant."""
    return rf"^{re.escape(base_name)}(\s*\(nist\).*|\.\d+)?$"

# --- Initial load ------------------------------------------------------------
def load_titles():
    a_path, c_path = find_file(ANTOINE_FILE), find_file(CORROSION_FILE)
    if not a_path or not c_path:
        error_msg = "Error: Ensure data files are in the search directory.\n\n"
        error_msg += "Expected one of:\n"
        error_msg += f"  {ANTOINE_FILE}.xlsx\n"
        error_msg += f"  {ANTOINE_FILE}.csv\n"
        error_msg += f"  {CORROSION_FILE}.xlsx\n"
        error_msg += f"  {CORROSION_FILE}.csv\n\n"
        error_msg += "Searched these folders:\n"
        for d in SEARCH_DIRS:
            error_msg += f"  {d}\n"
        return [], None, None, None, None, error_msg

    a_cols, c_cols = read_columns(a_path), read_columns(c_path)
    compounds = sorted(unique_compounds(a_cols) | unique_compounds(c_cols))
    return compounds, a_path, c_path, a_cols, c_cols, None

# --- Antoine maths -----------------------------------------------------------
def antoine_pressure(A, B, C, T):
    """Vapor pressure from Antoine parameters (same units as the source data)."""
    return 10 ** (A - (B / (T + C)))

def parse_parameter_sets(df):
    """Yield (col_name, A, B, C, T_min, T_max) for every valid column."""
    for col in df.columns:
        data = df[col].dropna().tolist()
        if len(data) >= ANTOINE_PARAM_COUNT:
            A, B, C, T1, T2 = (float(x) for x in data[:ANTOINE_PARAM_COUNT])
            yield col, A, B, C, T1, T2

# --- Main Window -------------------------------------------------------------
class ChemicalDashboard(QMainWindow):
    def __init__(self, compounds, a_path, c_path, a_cols, c_cols):
        super().__init__()
        self.compounds = compounds
        self.a_path = a_path
        self.c_path = c_path
        self.a_cols = a_cols
        self.c_cols = c_cols
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Chemical Properties Consolidated Dashboard')
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # --- Top Controls Area ---
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout()
        
        compound_label = QLabel("Compound:")
        self.compound_combo = QComboBox()
        self.compound_combo.addItems(self.compounds)
        self.compound_combo.setEditable(True)
        self.compound_combo.setMinimumWidth(300)
        self.compound_combo.currentTextChanged.connect(self.update_dashboard)
        
        temp_label = QLabel("Temperature (K):")
        
        self.temp_spinbox = QDoubleSpinBox()
        self.temp_spinbox.setMinimum(100.0)
        self.temp_spinbox.setMaximum(1000.0)
        self.temp_spinbox.setValue(DEFAULT_TEMP_K)
        self.temp_spinbox.setDecimals(1)
        self.temp_spinbox.setSingleStep(1.0)
        self.temp_spinbox.setMinimumWidth(100)
        self.temp_spinbox.valueChanged.connect(self.on_spinbox_changed)
        
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setMinimum(1000)  # 100.0 * 10
        self.temp_slider.setMaximum(10000) # 1000.0 * 10
        self.temp_slider.setValue(int(DEFAULT_TEMP_K * 10))
        self.temp_slider.setTickPosition(QSlider.TicksBelow)
        self.temp_slider.setTickInterval(1000)
        self.temp_slider.setMinimumWidth(300)
        self.temp_slider.valueChanged.connect(self.on_slider_changed)
        
        controls_layout.addWidget(compound_label)
        controls_layout.addWidget(self.compound_combo)
        controls_layout.addWidget(temp_label)
        controls_layout.addWidget(self.temp_spinbox)
        controls_layout.addWidget(self.temp_slider)
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)
        
        # --- Horizontal Layout Splitter (Left vs Right Panel) ---
        main_splitter = QSplitter(Qt.Horizontal)
        
        # ==========================================
        # LEFT PANEL: Graphs, Antoine Data, Log Text
        # ==========================================
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Vertical splitter inside left block to split graph and table comfortably
        left_inner_splitter = QSplitter(Qt.Vertical)
        
        # Plot Frame Area
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)  # Fixed internal caching canvas object
        plot_layout.addWidget(self.canvas)
        left_inner_splitter.addWidget(plot_widget)
        
        # Antoine Parameter Table Area
        antoine_container = QWidget()
        antoine_layout = QVBoxLayout(antoine_container)
        antoine_layout.setContentsMargins(0, 5, 0, 0)
        antoine_layout.addWidget(QLabel("<b>Antoine Parameters</b>"))
        self.antoine_table = QTableWidget()
        antoine_layout.addWidget(self.antoine_table)
        left_inner_splitter.addWidget(antoine_container)
        
        # Add internal components to left layout block
        left_layout.addWidget(left_inner_splitter)
        
        # Embedded System Logs Box placed at bottom left
        left_layout.addWidget(QLabel("<b>System Notifications & Status</b>"))
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(120)
        left_layout.addWidget(self.info_text)
        
        main_splitter.addWidget(left_container)
        
        # ==========================================
        # RIGHT PANEL: Corrosion Rectangle Block
        # ==========================================
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        right_layout.addWidget(QLabel("<b>Corrosion Matrix Data</b>"))
        self.corrosion_table = QTableWidget()
        right_layout.addWidget(self.corrosion_table)
        
        main_splitter.addWidget(right_container)
        
        # Default sizing allocation (e.g., 65% width to graph/parameters, 35% to corrosion)
        main_splitter.setSizes([900, 500])
        left_inner_splitter.setSizes([450, 250])
        
        main_layout.addWidget(main_splitter)
        
        # Initial update
        self.update_dashboard()
        
    def on_slider_changed(self, value):
        """Handle slider value changes and update spinbox."""
        temp_value = value / 10.0  # Convert back to actual temperature
        self.temp_spinbox.blockSignals(True)
        self.temp_spinbox.setValue(temp_value)
        self.temp_spinbox.blockSignals(False)
        self.update_dashboard()
        
    def on_spinbox_changed(self, value):
        """Handle spinbox value changes and update slider."""
        slider_value = int(value * 10)  # Convert to slider scale
        self.temp_slider.blockSignals(True)
        self.temp_slider.setValue(slider_value)
        self.temp_slider.blockSignals(False)
        self.update_dashboard()
        
    def update_dashboard(self):
        """Update all dashboard components when compound or temperature changes."""
        base_name = self.compound_combo.currentText()
        if not base_name:
            return
            
        curr_t = self.temp_spinbox.value()
        pattern = compound_pattern(base_name)
        
        try:
            self.render_antoine(base_name, pattern, curr_t)
            self.render_corrosion(base_name, pattern)
        except Exception as e:
            self.info_text.append(f"Error: {str(e)}")
            
    def render_antoine(self, base_name, pattern, curr_t):
        """Render Antoine pressure curves and data."""
        df = read_matching(self.a_path, self.a_cols, pattern)
        self.info_text.clear()
        
        if df is None or df.empty:
            self.info_text.append(f"No Antoine parameter data found for {base_name}.")
            self.ax.clear()
            self.canvas.draw()
            self.antoine_table.setRowCount(0)
            self.antoine_table.setColumnCount(0)
            return
            
        sets = list(parse_parameter_sets(df))
        
        # Update slider and spinbox bounds dynamically
        if sets:
            new_min = min(s[4] for s in sets)
            new_max = max(s[5] for s in sets)
            
            # Safeguard edge case scenario where parameter profile limits evaluate flat
            if new_min >= new_max:
                new_max = new_min + 100.0
                
            self.temp_spinbox.blockSignals(True)
            self.temp_spinbox.setMinimum(new_min)
            self.temp_spinbox.setMaximum(new_max)
            self.temp_spinbox.setValue(max(new_min, min(curr_t, new_max)))
            self.temp_spinbox.blockSignals(False)
            
            self.temp_slider.blockSignals(True)
            self.temp_slider.setMinimum(int(new_min * 10))
            self.temp_slider.setMaximum(int(new_max * 10))
            self.temp_slider.setValue(int(self.temp_spinbox.value() * 10))
            self.temp_slider.blockSignals(False)
            
            curr_t = self.temp_spinbox.value()
            
        self.ax.clear()
        
        valid = 0
        for col, A, B, C, T1, T2 in sets:
            tr = np.linspace(T1, T2, 100)
            self.ax.plot(tr, antoine_pressure(A, B, C, tr), label=str(col))
            if T1 <= curr_t <= T2:
                p = antoine_pressure(A, B, C, curr_t)
                self.info_text.append(f"Parameter set {col}: P={p:.4f} Bar")
                valid += 1
                
        if valid == 0:
            self.info_text.append(f"No valid parameter sets at {curr_t:.1f} K")
            
        self.ax.axvline(curr_t, color='red', linestyle='--', alpha=0.6,
                        label=f'Current: {curr_t:.1f}K')
        self.ax.set_title(f"Combined Vapor Pressure Curves for: {base_name}")
        self.ax.set_xlabel("Temperature (K)")
        
        # Applied proper scaling fixes instead of conflicting plain linear display
        #self.ax.set_yscale('log')
        self.ax.set_ylabel("Pressure (Bar)")
        
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax.grid(True, which="both", alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
        
        self.display_dataframe(df, self.antoine_table)
        
    def render_corrosion(self, base_name, pattern):
        """Render corrosion data table."""
        df = read_matching(self.c_path, self.c_cols, pattern)
        
        if df is None or df.empty:
            self.info_text.append(f"\nNo corrosion matrix entry for {base_name}.")
            self.corrosion_table.setRowCount(0)
            self.corrosion_table.setColumnCount(0)
        else:
            self.display_dataframe(df, self.corrosion_table)
            
    def display_dataframe(self, df, table_widget):
        """Display a pandas DataFrame in a QTableWidget."""
        table_widget.setRowCount(df.shape[0])
        table_widget.setColumnCount(df.shape[1])
        
        table_widget.setHorizontalHeaderLabels([str(col) for col in df.columns])
        table_widget.setVerticalHeaderLabels([str(idx) for idx in df.index])
        
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                value = df.iloc[i, j]
                item = QTableWidgetItem(str(value))
                table_widget.setItem(i, j, item)
                
        table_widget.resizeColumnsToContents()

# --- Entry point -------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    compounds, a_path, c_path, a_cols, c_cols, error_msg = load_titles()
    
    if error_msg:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Data Files Not Found")
        msg.setText(error_msg)
        msg.exec_()
        sys.exit(1)
        
    if not compounds:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("No Data")
        msg.setText("No compounds found in data files.")
        msg.exec_()
        sys.exit(1)
        
    window = ChemicalDashboard(compounds, a_path, c_path, a_cols, c_cols)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()