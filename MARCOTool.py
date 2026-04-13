import sys
import io
import os
import contextlib
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QGridLayout, QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QSlider, QCheckBox, QDoubleSpinBox, QHBoxLayout,
    QFrame, QHeaderView, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QColor, QBrush

from src.estimator import marcot_hr_estimator
from src.utils import print_results, snr_cal, multi_criteria, tables

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLIDER_SCALE = int(round(1 / 0.1))   # Maps float step 0.1 → integer 10
SPIN_DECIMALS = len(str(0.1).split(".")[-1]) if "." in str(0.1) else 0
RESULTS_CSV = "data/results_marcot.csv"
TOPSIS_CSV  = "data/score_total_st_TOPSIS.csv"
FIGURES_DIR = "Figures"

INITIAL_CRITERIA = [
    ("Reduction cost factor",                   "benefit"),
    ("Weight supported by the mount (kg)",      "cost"),
    ("Selected commercial output core (microns)", "cost"),
    ("Expected efficiency",                     "benefit"),
    ("Resolution with commercial fibers",       "benefit"),
    ("SNR fraction",                            "benefit"),
    ("Number of OTA for high efficiency",       "cost"),
]

TOPSIS_CRITERIA_COLS = [
    "OTA diameter (m)",
    "Reduction cost factor",
    "Weight supported by the mount (kg)",
    "Selected commercial output core (microns)",
    "Expected efficiency",
    "Resolution with commercial fibers",
    "SNR fraction",
    "Number of OTA for high efficiency",
]

TABLE_HEADER_STYLE_GREEN = (
    "QHeaderView::section { background-color: #4CAF50; color: white; font-weight: bold; }"
)
TABLE_HEADER_STYLE_BLUE = (
    "QHeaderView::section { background-color: #2196F3; color: white; font-weight: bold; }"
)


# ---------------------------------------------------------------------------
# Helper: colour a QTableWidget row
# ---------------------------------------------------------------------------

def _set_table_row(table, row_index, col0_text, col1_text, even_color, odd_color):
    """Insert styled items into two columns of *table* at *row_index*."""
    bg = QBrush(QColor(*(even_color if row_index % 2 == 0 else odd_color)))
    fg = QBrush(QColor(0, 0, 0))

    for col, text in enumerate((col0_text, col1_text)):
        item = QTableWidgetItem(text)
        item.setForeground(fg)
        item.setBackground(bg)
        table.setItem(row_index, col, item)


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class MarcotApp(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MARCOTool v1.9")

        self.graph_images: list[str] = []
        self.plot_index: int = 0

        layout = QVBoxLayout()
        layout.addWidget(self._build_logo())
        layout.addWidget(self._build_input_tabs())
        layout.addWidget(self._build_advanced_section())
        layout.addWidget(self._build_run_button())
        layout.addWidget(self._build_output_tabs())
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------

    def _build_logo(self) -> QLabel:
        logo_label = QLabel(self)
        pixmap = QPixmap("Images/logo.png")
        if pixmap.isNull():
            logo_label.setText("Loading logo has failed")
        else:
            pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
        return logo_label

    def _build_input_tabs(self) -> QTabWidget:
        self.input_tabs = QTabWidget()
        self.input_tabs.addTab(self._build_basic_params_tab(),    "Basic parameters")
        self.input_tabs.addTab(self._build_detector_params_tab(), "Detector parameters")
        self.input_tabs.addTab(self._build_criteria_tab(),        "Criteria")
        return self.input_tabs

    def _build_basic_params_tab(self) -> QScrollArea:
        self._init_basic_widgets()

        grid = QGridLayout()
        row = 0

        def add(label, *widgets):
            nonlocal row
            grid.addWidget(QLabel(label), row, 0)
            for col, w in enumerate(widgets, start=1):
                grid.addWidget(w, row, col)
            row += 1

        add("Module diameter (m)",        self.module_diam_slider,       self.module_diam)
        add("F/# output",                 self.f_number_out_spin,        self.f_number_out_locked)
        add("Core diameter output (um)",  self.d_core_out_spin,          self.d_core_out_locked)
        add("Total effective aperture (m)", self.telescope_aperture_slider, self.telescope_aperture)
        add("Focal Adapter (F/# input)",  self.focal_adapter_spin,       self.focal_adapter_locked)
        add("Seeing FWHM (arcsec)",       self.seeing_spin)
        add("Sky Aperture (arcsec)",      self.sky_aperture_spin,        self.sky_aperture_locked)

        grid.addWidget(self.tiptilt_checkbox,  row, 0)
        grid.addWidget(self.PL_effi_checkbox,  row, 1)
        row += 1

        add("Encircled Energy",           self.encircled_spin)
        add("Minimum wavelength (nm)",    self.lambda_min_spin)
        add("Maximum wavelength (nm)",    self.lambda_max_spin)

        grid.addWidget(self.pseudoslit_checkbox, row, 0)
        grid.addWidget(self.superpl_checkbox,    row, 1)

        widget = QWidget()
        widget.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _build_detector_params_tab(self) -> QScrollArea:
        self._init_detector_widgets()

        grid = QGridLayout()
        row = 0

        def add(label, widget):
            nonlocal row
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget,        row, 1)
            row += 1

        add("Plate Scale (um/arcsec)", self.plate_scale_spin)
        add("Read noise (e-)",         self.R_noise_spin)
        add("Gain (e-/ADU)",           self.g_spin)
        add("Pixel size (um/px)",      self.pixel_size_um_spin)
        add("DARK · 1e3 (e-/s)",       self.DARK_spin)
        add("Quantum Efficiency",      self.QE_spin)

        widget = QWidget()
        widget.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _build_criteria_tab(self) -> QWidget:
        self.criteria_table = QTableWidget()
        self.criteria_table.setColumnCount(3)
        self.criteria_table.setHorizontalHeaderLabels(["Criteria", "Benefit", "Cost"])
        self.criteria_table.setRowCount(len(INITIAL_CRITERIA))

        header = self.criteria_table.horizontalHeader()
        for col in range(3):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)

        for i, (name, kind) in enumerate(INITIAL_CRITERIA):
            self.criteria_table.setItem(i, 0, QTableWidgetItem(name))
            self._add_benefit_cost_checkboxes(i, default_benefit=(kind == "benefit"))

        self.btn_add_crit = QPushButton("Add criteria")
        self.btn_del_crit = QPushButton("Remove criteria")
        self.btn_add_crit.clicked.connect(self.add_criteria_row)
        self.btn_del_crit.clicked.connect(self.del_criteria_row)

        layout = QVBoxLayout()
        layout.addWidget(self.criteria_table)
        layout.addWidget(self.btn_add_crit)
        layout.addWidget(self.btn_del_crit)

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _build_advanced_section(self) -> QWidget:
        """Returns the collapsible advanced spectrograph parameters block."""
        self.advanced_button = QPushButton("Show spectrograph parameters")
        self.advanced_button.setCheckable(True)
        self.advanced_button.toggled.connect(self._toggle_advanced_params)

        self._init_advanced_widgets()

        grid = QGridLayout()
        grid.addWidget(QLabel("Grooves per mm"),               0, 0)
        grid.addWidget(self.grooves_mm_spin,                   0, 1)
        grid.addWidget(self.grooves_mm_locked,                 0, 2)
        grid.addWidget(self.resolution_label,                  1, 0)
        grid.addWidget(self.resolution_slider,                 1, 1)
        grid.addWidget(self.resolution_locked,                 1, 2)
        grid.addWidget(QLabel("Beam diameter (mm)"),           2, 0)
        grid.addWidget(self.beam_size_mm_spin,                 2, 1)
        grid.addWidget(self.beam_size_locked,                  2, 2)
        grid.addWidget(QLabel("Resolution element (px)"),      3, 0)
        grid.addWidget(self.rel_element_spin,                  3, 1)
        grid.addWidget(self.rel_element_locked,                3, 2)
        grid.addWidget(QLabel("Focal length camera (mm)"),     4, 0)
        grid.addWidget(self.f_cam_mm_spin,                     4, 1)
        grid.addWidget(self.f_cam_mm_locked,                   4, 2)
        grid.addWidget(QLabel("Focal length collimator (mm)"), 5, 0)
        grid.addWidget(self.f_coll_mm_spin,                    5, 1)
        grid.addWidget(QLabel("Incident Angle on Grating (deg)"), 6, 0)
        grid.addWidget(self.incident_angle_spin,               6, 1)
        grid.addWidget(self.slicer_checkbox,                   7, 0)
        grid.addWidget(self.nir_arm_checkbox,                  7, 1)
        grid.addWidget(self.echelle_checkbox,                  7, 2)

        self.advanced_frame = QFrame()
        self.advanced_frame.setFrameShape(QFrame.StyledPanel)
        self.advanced_frame.setLayout(grid)
        self.advanced_frame.hide()

        container = QWidget()
        vbox = QVBoxLayout()
        vbox.addWidget(self.advanced_button)
        vbox.addWidget(self.advanced_frame)
        container.setLayout(vbox)
        return container

    def _build_run_button(self) -> QPushButton:
        btn = QPushButton("Run")
        btn.clicked.connect(self.run_calculations)
        return btn

    def _build_output_tabs(self) -> QTabWidget:
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.results_table.horizontalHeader().setStyleSheet(TABLE_HEADER_STYLE_GREEN)

        # Best set-up table
        self.best_table = QTableWidget()
        self.best_table.setColumnCount(2)
        self.best_table.setHorizontalHeaderLabels(["Criteria", "Value"])
        self.best_table.horizontalHeader().setStyleSheet(TABLE_HEADER_STYLE_BLUE)

        # Text output
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)

        # Graph tab
        self.graph_canvas = FigureCanvas(plt.figure(figsize=(5, 4)))
        self.prev_button = QPushButton("←")
        self.next_button = QPushButton("→")
        self.prev_button.clicked.connect(self.show_prev_plot)
        self.next_button.clicked.connect(self.show_next_plot)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)

        graph_widget = QWidget()
        graph_layout = QVBoxLayout()
        graph_layout.addWidget(self.graph_canvas)
        graph_layout.addLayout(nav_layout)
        graph_widget.setLayout(graph_layout)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.results_table, "Summary table")
        self.tabs.addTab(graph_widget,       "Plots")
        self.tabs.addTab(self.text_output,   "Text details")
        self.tabs.addTab(self.best_table,    "Best set-up")
        return self.tabs

    # ------------------------------------------------------------------
    # Widget initialisation helpers
    # ------------------------------------------------------------------

    def _make_slider(self, min_val, max_val, default):
        s = QSlider(Qt.Horizontal)
        s.setMinimum(int(min_val * SLIDER_SCALE))
        s.setMaximum(int(max_val * SLIDER_SCALE))
        s.setValue(int(default * SLIDER_SCALE))
        s.setSingleStep(1)
        return s

    def _make_spin(self, min_val, max_val, default, step=0.1, decimals=None):
        sp = QDoubleSpinBox()
        sp.setRange(min_val, max_val)
        sp.setSingleStep(step)
        sp.setValue(default)
        if decimals is not None:
            sp.setDecimals(decimals)
        else:
            sp.setDecimals(SPIN_DECIMALS)
        return sp

    def _init_basic_widgets(self):
        # Module diameter slider + spin (synced)
        self.module_diam_slider = self._make_slider(0, 15, 5)
        self.module_diam = self._make_spin(0, 15, 5)
        self.module_diam_slider.valueChanged.connect(self._on_slider_module)
        self.module_diam.valueChanged.connect(self._on_spin_module)

        self.f_number_out_spin  = self._make_spin(1, 15, 3.5)
        self.f_number_out_locked = QCheckBox("Lock")
        self.f_number_out_locked.setChecked(True)

        self.d_core_out_spin    = self._make_spin(1, 1000, 100)
        self.d_core_out_locked  = QCheckBox("Lock")
        self.d_core_out_locked.setChecked(False)

        # Telescope aperture slider + spin (synced)
        self.telescope_aperture_slider = self._make_slider(0, 40, 15)
        self.telescope_aperture = self._make_spin(0, 40, 15)
        self.telescope_aperture_slider.valueChanged.connect(self._on_slider_telescope)
        self.telescope_aperture.valueChanged.connect(self._on_spin_telescope)

        self.focal_adapter_spin   = self._make_spin(0.1, 15, 5, step=0.05)
        self.focal_adapter_locked = QCheckBox("Lock")
        self.focal_adapter_locked.setChecked(True)

        self.seeing_spin = self._make_spin(0.1, 2.0, 1.00, step=0.05)

        self.sky_aperture_spin   = self._make_spin(0, 10, 1.5)
        self.sky_aperture_locked = QCheckBox("Lock")
        self.sky_aperture_locked.setChecked(True)

        self.tiptilt_checkbox  = QCheckBox("Use Tip/Tilt")
        self.PL_effi_checkbox  = QCheckBox("Maximize PL Efficiency")
        self.PL_effi_checkbox.setChecked(False)

        self.encircled_spin    = self._make_spin(0.5, 1.0, 0.95, step=0.01, decimals=2)
        self.lambda_min_spin   = self._make_spin(300, 1500, 550, step=10, decimals=0)
        self.lambda_max_spin   = self._make_spin(350, 2000, 1050, step=10, decimals=0)

        self.pseudoslit_checkbox = QCheckBox("Use Pseudoslit")
        self.pseudoslit_checkbox.setChecked(True)
        self.pseudoslit_checkbox.toggled.connect(self._sync_pseudoslit)

        self.superpl_checkbox = QCheckBox("Use Photonic Lantern")
        self.superpl_checkbox.setChecked(False)
        self.superpl_checkbox.toggled.connect(self._sync_superpl)

    def _init_detector_widgets(self):
        self.plate_scale_spin   = self._make_spin(0, 500, 169, step=1, decimals=0)
        self.R_noise_spin       = self._make_spin(0, 10.0, 5, step=0.01, decimals=2)
        self.g_spin             = self._make_spin(0, 10.0, 1, step=0.01, decimals=2)
        self.pixel_size_um_spin = self._make_spin(1.0, 50.0, 15, step=0.5)
        self.DARK_spin          = self._make_spin(0, 10000, 3000, step=1, decimals=0)
        self.QE_spin            = self._make_spin(0, 1.0, 0.95, step=0.01, decimals=2)

    def _init_advanced_widgets(self):
        self.grooves_mm_spin   = self._make_spin(10, 5000, 31.6, step=0.5)
        self.grooves_mm_locked = QCheckBox("Lock")
        self.grooves_mm_locked.setChecked(True)

        self.resolution_label  = QLabel("Resolving power: 82000")
        self.resolution_slider = QSlider(Qt.Horizontal)
        self.resolution_slider.setMinimum(1)
        self.resolution_slider.setMaximum(200000)
        self.resolution_slider.setValue(82000)
        self.resolution_slider.setTickInterval(1)
        self.resolution_slider.valueChanged.connect(self._update_resolution_label)
        self.resolution_locked = QCheckBox("Lock")
        self.resolution_locked.setChecked(True)

        self.echelle_checkbox  = QCheckBox("Echelle")
        self.echelle_checkbox.setChecked(True)

        self.beam_size_mm_spin = self._make_spin(50, 300, 154.8, step=1)
        self.beam_size_locked  = QCheckBox("Lock")
        self.beam_size_locked.setChecked(False)

        self.rel_element_spin   = self._make_spin(1.0, 5.0, 2.8, step=0.1)
        self.rel_element_locked = QCheckBox("Lock")
        self.rel_element_locked.setChecked(True)

        self.f_cam_mm_spin   = self._make_spin(1, 1000, 455, step=1, decimals=0)
        self.f_cam_mm_locked = QCheckBox("Lock")
        self.f_cam_mm_locked.setChecked(True)

        self.f_coll_mm_spin      = self._make_spin(1, 1000, 536.07, step=0.1)
        self.incident_angle_spin = self._make_spin(0, 360, 75.2, step=0.1)

        self.slicer_checkbox  = QCheckBox("Use of slicer")
        self.slicer_checkbox.setChecked(True)
        self.nir_arm_checkbox = QCheckBox("NIR channel")

    # ------------------------------------------------------------------
    # Criteria table helpers
    # ------------------------------------------------------------------

    def _add_benefit_cost_checkboxes(self, row: int, default_benefit: bool = True):
        """Add mutually exclusive Benefit/Cost checkboxes to *row*."""
        benefit_cb = QCheckBox()
        cost_cb    = QCheckBox()
        benefit_cb.setChecked(default_benefit)
        cost_cb.setChecked(not default_benefit)

        benefit_cb.stateChanged.connect(lambda s, cb=cost_cb:    cb.setChecked(False) if s == Qt.Checked else None)
        cost_cb.stateChanged.connect(   lambda s, cb=benefit_cb: cb.setChecked(False) if s == Qt.Checked else None)

        self.criteria_table.setCellWidget(row, 1, benefit_cb)
        self.criteria_table.setCellWidget(row, 2, cost_cb)

    def add_criteria_row(self):
        row = self.criteria_table.rowCount()
        self.criteria_table.insertRow(row)
        self.criteria_table.setItem(row, 0, QTableWidgetItem("New criteria"))
        self._add_benefit_cost_checkboxes(row, default_benefit=True)

    def del_criteria_row(self):
        row = self.criteria_table.currentRow()
        if row >= 0:
            self.criteria_table.removeRow(row)

    def get_criteria_dict(self) -> dict:
        criteria = {}
        for row in range(self.criteria_table.rowCount()):
            name_item      = self.criteria_table.item(row, 0)
            benefit_widget = self.criteria_table.cellWidget(row, 1)
            cost_widget    = self.criteria_table.cellWidget(row, 2)

            if benefit_widget.isChecked():
                kind = "benefit"
            elif cost_widget.isChecked():
                kind = "cost"
            else:
                kind = "none"

            criteria[name_item.text()] = {"type": kind}
        return criteria

    # ------------------------------------------------------------------
    # Slider / spin synchronisation
    # ------------------------------------------------------------------

    def _on_slider_module(self, ivalue: int):
        fvalue = ivalue / SLIDER_SCALE
        if abs(self.module_diam.value() - fvalue) > 1e-12:
            self.module_diam.blockSignals(True)
            self.module_diam.setValue(fvalue)
            self.module_diam.blockSignals(False)

    def _on_spin_module(self, fvalue: float):
        ivalue = int(round(fvalue * SLIDER_SCALE))
        if self.module_diam_slider.value() != ivalue:
            self.module_diam_slider.blockSignals(True)
            self.module_diam_slider.setValue(ivalue)
            self.module_diam_slider.blockSignals(False)

    def _on_slider_telescope(self, ivalue: int):
        fvalue = ivalue / SLIDER_SCALE
        if abs(self.telescope_aperture.value() - fvalue) > 1e-12:
            self.telescope_aperture.blockSignals(True)
            self.telescope_aperture.setValue(fvalue)
            self.telescope_aperture.blockSignals(False)

    def _on_spin_telescope(self, fvalue: float):
        ivalue = int(round(fvalue * SLIDER_SCALE))
        if self.telescope_aperture_slider.value() != ivalue:
            self.telescope_aperture_slider.blockSignals(True)
            self.telescope_aperture_slider.setValue(ivalue)
            self.telescope_aperture_slider.blockSignals(False)

    # ------------------------------------------------------------------
    # Checkbox synchronisation (Pseudoslit ↔ Photonic Lantern)
    # ------------------------------------------------------------------

    def _sync_superpl(self, checked: bool):
        if checked:
            self.pseudoslit_checkbox.blockSignals(True)
            self.pseudoslit_checkbox.setChecked(False)
            self.pseudoslit_checkbox.blockSignals(False)

    def _sync_pseudoslit(self, checked: bool):
        if checked:
            self.superpl_checkbox.blockSignals(True)
            self.superpl_checkbox.setChecked(False)
            self.superpl_checkbox.blockSignals(False)

    # ------------------------------------------------------------------
    # Misc UI callbacks
    # ------------------------------------------------------------------

    def _toggle_advanced_params(self, checked: bool):
        if checked:
            self.advanced_frame.show()
            self.advanced_button.setText("Hide spectrograph parameters")
        else:
            self.advanced_frame.hide()
            self.advanced_button.setText("Show spectrograph parameters")

    def _update_resolution_label(self):
        self.resolution_label.setText(f"Resolving power: {self.resolution_slider.value()}")

    # ------------------------------------------------------------------
    # Plot navigation
    # ------------------------------------------------------------------

    def show_plot_at_index(self, index: int):
        if 0 <= index < len(self.graph_images):
            self.graph_canvas.figure.clear()
            ax = self.graph_canvas.figure.add_subplot(111)
            ax.imshow(plt.imread(self.graph_images[index]))
            ax.axis("off")
            ax.set_title(f"Plot {index + 1} / {len(self.graph_images)}")
            self.graph_canvas.draw()

    def show_prev_plot(self):
        if self.plot_index > 0:
            self.plot_index -= 1
            self.show_plot_at_index(self.plot_index)

    def show_next_plot(self):
        if self.plot_index < len(self.graph_images) - 1:
            self.plot_index += 1
            self.show_plot_at_index(self.plot_index)

    # ------------------------------------------------------------------
    # Core calculation
    # ------------------------------------------------------------------

    def _collect_params(self) -> list:
        """Read all widget values and return them as an ordered list."""
        return [
            float(self.module_diam.value()),
            float(self.f_number_out_spin.value()),
            int(self.f_number_out_locked.isChecked()),
            float(self.d_core_out_spin.value()),
            int(self.d_core_out_locked.isChecked()),
            float(self.telescope_aperture.value()),
            float(self.focal_adapter_spin.value()),
            int(self.focal_adapter_locked.isChecked()),
            float(self.seeing_spin.value()),
            float(self.sky_aperture_spin.value()),
            int(self.sky_aperture_locked.isChecked()),
            self.tiptilt_checkbox.isChecked(),
            self.PL_effi_checkbox.isChecked(),
            float(self.encircled_spin.value()),
            int(self.lambda_min_spin.value()),
            int(self.lambda_max_spin.value()),
            self.pseudoslit_checkbox.isChecked(),
            self.superpl_checkbox.isChecked(),
            float(self.grooves_mm_spin.value()),
            float(self.resolution_slider.value()),
            float(self.beam_size_mm_spin.value()),
            int(self.beam_size_locked.isChecked()),
            float(self.pixel_size_um_spin.value()),
            float(self.rel_element_spin.value()),
            self.slicer_checkbox.isChecked(),
            float(self.f_cam_mm_spin.value()),
            int(self.f_cam_mm_locked.isChecked()),
            float(self.f_coll_mm_spin.value()),
            float(self.incident_angle_spin.value()),
            int(self.echelle_checkbox.isChecked()),
            self.nir_arm_checkbox.isChecked(),
        ]

    def _save_results_csv(self, results: dict) -> int:
        """Serialize *results* to TSV and return the number of rows written."""
        def is_scalar(x):
            return isinstance(x, (int, float, np.integer, np.floating))

        lengths = [
            v.size if isinstance(v, np.ndarray) else len(v)
            for v in results.values()
            if isinstance(v, (np.ndarray, list, tuple))
        ]
        target_len = max(lengths) if lengths else 1

        cols = {}
        for k, v in results.items():
            if k == "" or v is None:
                continue
            if isinstance(v, np.ndarray):
                cols[k] = pd.Series(np.asarray(v).ravel())
            elif isinstance(v, (list, tuple)):
                cols[k] = pd.Series(list(v))
            elif is_scalar(v):
                cols[k] = pd.Series([float(v)] * target_len)

        df = pd.DataFrame(cols).reindex(range(target_len))
        df.to_csv(RESULTS_CSV, sep="\t", index=False)
        print("\033[1;4;32mFile 'results_marcot.csv' was saved successfully\033[0m")
        return target_len

    def _update_results_table(self, results: dict):
        """Populate the summary results table."""
        self.results_table.setRowCount(len(results))
        for i, (key, value) in enumerate(results.items()):
            val_str = (
                np.array2string(np.asarray(value), precision=3)
                if isinstance(value, (list, np.ndarray, np.generic))
                else str(value)
            )
            _set_table_row(
                self.results_table, i,
                str(key), val_str,
                even_color=(240, 240, 240),
                odd_color=(255, 255, 255),
            )
        self.results_table.resizeColumnsToContents()

    def _update_best_table(self):
        """Populate the Best set-up table from the TOPSIS scores CSV."""
        if not os.path.exists(TOPSIS_CSV):
            return

        df = pd.read_csv(TOPSIS_CSV, sep="\t")

        # Parse any list-like string columns
        for col in df.columns:
            if df[col].astype(str).str.startswith("[").any():
                df[col] = df[col].apply(
                    lambda x: np.fromstring(x.strip("[] "), sep=" ")
                    if isinstance(x, str) and "[" in x else x
                )

        # Find the best alternative index from the first (and only) row
        row = df.iloc[0]
        score_total = row["score_total_st_TOPSIS"]
        if isinstance(score_total, str) and score_total.startswith("["):
            score_array = np.fromstring(score_total.strip("[] "), sep=" ")
        elif isinstance(score_total, (np.ndarray, list)):
            score_array = np.asarray(score_total)
        else:
            score_array = np.array([score_total])

        best_index = int(np.argmax(score_array))
        score_val  = score_array[best_index]

        self.best_table.setRowCount(0)
        for i, criterio in enumerate(TOPSIS_CRITERIA_COLS[1:]):   # skip OTA diameter
            val = row[criterio]
            if isinstance(val, str) and val.startswith("["):
                arr = np.fromstring(val.strip("[] "), sep=" ")
                display = arr[best_index] if len(arr) > best_index else arr[0]
            elif isinstance(val, (np.ndarray, list)):
                arr = np.asarray(val)
                display = arr[best_index] if len(arr) > best_index else arr[0]
            else:
                display = val

            self.best_table.insertRow(i)
            _set_table_row(
                self.best_table, i,
                str(criterio), str(display),
                even_color=(225, 245, 255),
                odd_color=(255, 255, 255),
            )

        # Append TOPSIS score row
        score_row = self.best_table.rowCount()
        self.best_table.insertRow(score_row)
        score_item = QTableWidgetItem(str(score_val))
        score_item.setForeground(QBrush(QColor(0, 0, 0)))
        score_item.setBackground(QBrush(QColor(255, 223, 186)))
        self.best_table.setItem(score_row, 0, QTableWidgetItem("score_total_st_TOPSIS"))
        self.best_table.setItem(score_row, 1, score_item)
        self.best_table.setSortingEnabled(True)
        self.best_table.resizeColumnsToContents()

    def run_calculations(self):
        try:
            results = marcot_hr_estimator(*self._collect_params())
            self._save_results_csv(results)
            self._update_results_table(results)

            # Build text output
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                print_results(results)
            output_text = buffer.getvalue()

            # SNR calculation
            params_detector = [
                float(self.plate_scale_spin.value()),
                float(self.R_noise_spin.value()),
                float(self.g_spin.value()),
                float(self.pixel_size_um_spin.value()),
                float(self.sky_aperture_spin.value()),
                float(self.DARK_spin.value()),
                float(self.QE_spin.value()),
                int(self.slicer_checkbox.isChecked()),
                int(self.superpl_checkbox.isChecked()),
            ]
            snr_val = float(np.asarray(snr_cal(RESULTS_CSV, *params_detector)).ravel()[0])
            output_text += f"\nSNR Fraction: {snr_val:.3f}\n"

            # Multi-criteria analysis
            criteria = self.get_criteria_dict()
            multi_criteria(RESULTS_CSV, criteria)
            output_text += "\nMultiple criteria calculated.\n"
            tables(criteria)

            self.text_output.setText(output_text)

            # Refresh plots
            self.graph_images = sorted(
                os.path.join(FIGURES_DIR, f)
                for f in os.listdir(FIGURES_DIR)
                if f.endswith(".png")
            )
            self.plot_index = 0
            if self.graph_images:
                self.show_plot_at_index(0)

            self._update_best_table()

        except Exception:
            QMessageBox.critical(self, "Error", traceback.format_exc())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MarcotApp()
    window.resize(900, 1000)
    window.show()
    sys.exit(app.exec_())
