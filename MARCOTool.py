import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QGridLayout, QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QSlider, QCheckBox, QDoubleSpinBox, QHBoxLayout, QFrame, QHeaderView, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QColor, QBrush
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from src.estimator import marcot_hr_estimator
from src.utils import print_results, snr_cal, multi_criteria, tables
import numpy as np
import io
import contextlib
import os
import pandas as pd
import traceback

class MarcotApp(QWidget):

    # --- Generate the check boxes for Benefict and Cost ---
    def add_benefit_cost_checkboxes(self, row):
        benefit_cb = QCheckBox()
        cost_cb = QCheckBox()

        benefit_cb.stateChanged.connect(lambda s: (s == Qt.Checked) and cost_cb.setChecked(False))
        cost_cb.stateChanged.connect(lambda s: (s == Qt.Checked) and benefit_cb.setChecked(False))

        self.criteria_table.setCellWidget(row, 2, benefit_cb)
        self.criteria_table.setCellWidget(row, 3, cost_cb)
        
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MARCOT HR Estimator v1.9")

        layout = QVBoxLayout()

        # Add logo
        logo_label = QLabel(self)
        pixmap = QPixmap("Images/logo.png")
        if pixmap.isNull():
            logo_label.setText("Loading logo has failed")
        else:
            pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        param_layout = QGridLayout()

        # Input parameters control
        self.module_diam_label = QLabel("Module diameter (m):")
        self.module_diam_slider = QSlider(Qt.Horizontal)
        self.module_diam_slider.setMinimum(1)
        self.module_diam_slider.setMaximum(15)
        self.module_diam_slider.setValue(5)
        self.module_diam_slider.setTickInterval(1)
        self.module_diam_slider.valueChanged.connect(self.update_module_diam_label)
        
        self.f_number_out_spin = QDoubleSpinBox()
        self.f_number_out_spin.setRange(1, 15)
        self.f_number_out_spin.setSingleStep(0.1)
        self.f_number_out_spin.setValue(3.5)
        self.f_number_out_locked = QCheckBox("Lock")
        self.f_number_out_locked.setChecked(True)
        
        self.d_core_out_spin = QDoubleSpinBox()
        self.d_core_out_spin.setRange(1, 1000)
        self.d_core_out_spin.setSingleStep(0.1)
        self.d_core_out_spin.setValue(100)
        self.d_core_out_locked = QCheckBox("Lock")
        self.d_core_out_locked.setChecked(False)
        
        self.telescope_aperture_label = QLabel("Telescope effective aperture (m):")
        self.telescope_aperture_slider = QSlider(Qt.Horizontal)
        self.telescope_aperture_slider.setMinimum(1)
        self.telescope_aperture_slider.setMaximum(30)
        self.telescope_aperture_slider.setValue(15)
        self.telescope_aperture_slider.setTickInterval(1)
        self.telescope_aperture_slider.valueChanged.connect(self.update_telescope_aperture_label)

        self.seeing_spin = QDoubleSpinBox()
        self.seeing_spin.setRange(0.1, 2.0)
        self.seeing_spin.setSingleStep(0.05)
        self.seeing_spin.setValue(1.0)
        
        self.sky_aperture_spin = QDoubleSpinBox()
        self.sky_aperture_spin.setRange(0, 10)
        self.sky_aperture_spin.setSingleStep(0.1)
        self.sky_aperture_spin.setValue(1.5)
        self.sky_aperture_locked = QCheckBox("Lock")
        self.sky_aperture_locked.setChecked(True)

        self.tiptilt_checkbox = QCheckBox("Use Tip/Tilt")

        self.encircled_spin = QDoubleSpinBox()
        self.encircled_spin.setRange(0.5, 1.0)
        self.encircled_spin.setSingleStep(0.01)
        self.encircled_spin.setValue(0.95)

        self.lambda_min_spin = QDoubleSpinBox()
        self.lambda_min_spin.setRange(300, 1500)
        self.lambda_min_spin.setSingleStep(10)
        self.lambda_min_spin.setValue(500)

        self.lambda_max_spin = QDoubleSpinBox()
        self.lambda_max_spin.setRange(500, 2000)
        self.lambda_max_spin.setSingleStep(10)
        self.lambda_max_spin.setValue(1000)

        self.pseudoslit_checkbox = QCheckBox("Use Pseudoslit")
        self.pseudoslit_checkbox.setChecked(True)

        self.superpl_checkbox = QCheckBox("Use Super-PL")

        # Set grid
        row = 0
        param_layout.addWidget(self.module_diam_label, row, 0)
        param_layout.addWidget(self.module_diam_slider, row, 1)
        row += 1
        
        param_layout.addWidget(QLabel("F/# output"), row, 0)
        param_layout.addWidget(self.f_number_out_spin, row, 1)
        param_layout.addWidget(self.f_number_out_locked, row, 2)
        row += 1
        
        param_layout.addWidget(QLabel("Core diameter output (um)"), row, 0)
        param_layout.addWidget(self.d_core_out_spin, row, 1)
        param_layout.addWidget(self.d_core_out_locked, row, 2)
        row += 1
        
        param_layout.addWidget(self.telescope_aperture_label, row, 0)
        param_layout.addWidget(self.telescope_aperture_slider, row, 1)
        row += 1

        param_layout.addWidget(QLabel("Seeing FWHM (arcsec)"), row, 0)
        param_layout.addWidget(self.seeing_spin, row, 1)
        row += 1
        
        param_layout.addWidget(QLabel("Sky Aperture (arcsec)"), row, 0)
        param_layout.addWidget(self.sky_aperture_spin, row, 1)
        param_layout.addWidget(self.sky_aperture_locked, row, 2)
        row += 1

        param_layout.addWidget(self.tiptilt_checkbox, row, 0)
        row += 1

        param_layout.addWidget(QLabel("Encircled Energy"), row, 0)
        param_layout.addWidget(self.encircled_spin, row, 1)
        row += 1

        param_layout.addWidget(QLabel("Minimun wavelength (nm)"), row, 0)
        param_layout.addWidget(self.lambda_min_spin, row, 1)
        row += 1

        param_layout.addWidget(QLabel("Maximun wavelength (nm)"), row, 0)
        param_layout.addWidget(self.lambda_max_spin, row, 1)
        row += 1

        param_layout.addWidget(self.pseudoslit_checkbox, row, 0)
        row += 1

        param_layout.addWidget(self.superpl_checkbox, row, 0)
        row += 1

        # --- Tabs ofr input parameters ---
        self.input_tabs = QTabWidget()

        # Tab 1: basic parameters
        basic_params_widget = QWidget()
        basic_params_widget.setLayout(param_layout)
        #self.input_tabs.addTab(basic_params_widget, "Basic parameters")
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(basic_params_widget)

        self.input_tabs.addTab(scroll, "Basic parameters")


        # Tab 2: criteria
        criteria_widget = QWidget()
        crit_layout = QVBoxLayout()

        self.criteria_table = QTableWidget()
        self.criteria_table.setColumnCount(3)
        self.criteria_table.setHorizontalHeaderLabels(["Criteria", "Benefit", "Cost"])
        self.criteria_table.setRowCount(7)

        initial_criteria = [
            ("Reduction cost factor", "benefit"),
            ("Weight supported by the mount (kg)", "cost"),
            ("Selected commercial output core (microns)", "cost"),
            ("Expected efficiency", "benefit"),
            ("Resolution with commercial fibers", "benefit"),
            ("SNR fraction", "benefit"),
            ("Number of OTA for high efficiency", "cost"),
        ]

        header = self.criteria_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        for i, (crit, type_str) in enumerate(initial_criteria):
            self.criteria_table.setItem(i, 0, QTableWidgetItem(crit))
                
            benefit_cb = QCheckBox()
            cost_cb = QCheckBox()
                
            if type_str.lower() == "benefit":
                benefit_cb.setChecked(True)
            else:
                cost_cb.setChecked(True)
                    
            benefit_cb.stateChanged.connect(lambda s, cb=cost_cb: (s == Qt.Checked) and cb.setChecked(False))
            cost_cb.stateChanged.connect(lambda s, cb=benefit_cb: (s == Qt.Checked) and cb.setChecked(False))
    
            self.criteria_table.setCellWidget(i, 1, benefit_cb)
            self.criteria_table.setCellWidget(i, 2, cost_cb)


        self.btn_add_crit = QPushButton("Add criteria")
        self.btn_del_crit = QPushButton("Remove criteria")
        self.btn_add_crit.clicked.connect(self.add_criteria_row)
        self.btn_del_crit.clicked.connect(self.del_criteria_row)

        crit_layout.addWidget(self.criteria_table)
        crit_layout.addWidget(self.btn_add_crit)
        crit_layout.addWidget(self.btn_del_crit)
        criteria_widget.setLayout(crit_layout)

        self.input_tabs.addTab(criteria_widget, "Criteria")

        layout.addWidget(self.input_tabs)

        # Bottom to show advanced spectrograph parameters
        self.advanced_button = QPushButton("Advanced parameters")
        self.advanced_button.setCheckable(True)
        self.advanced_button.toggled.connect(self.toggle_advanced_params)
        layout.addWidget(self.advanced_button)

        # Hidden advanced parameters
        self.advanced_frame = QFrame()
        self.advanced_frame.setFrameShape(QFrame.StyledPanel)
        self.advanced_frame.hide()

        advanced_layout = QGridLayout()
        
        self.grooves_mm_spin = QDoubleSpinBox()
        self.grooves_mm_spin.setRange(10, 100)
        self.grooves_mm_spin.setSingleStep(0.5)
        self.grooves_mm_spin.setValue(31.6)
        self.grooves_mm_locked = QCheckBox("Lock")
        self.grooves_mm_locked.setChecked(True)
        
        self.resolution_label = QLabel("Resolving power:")
        self.resolution_slider = QSlider(Qt.Horizontal)
        self.resolution_slider.setMinimum(1)
        self.resolution_slider.setMaximum(200000)
        self.resolution_slider.setValue(100000)
        self.resolution_slider.setTickInterval(1)
        self.resolution_slider.valueChanged.connect(self.update_resolution_label)
        self.resolution_locked = QCheckBox("Lock")
        self.resolution_locked.setChecked(True)

        self.mag_factor_spin = QDoubleSpinBox()
        self.mag_factor_spin.setRange(0.5, 3.0)
        self.mag_factor_spin.setSingleStep(0.05)
        self.mag_factor_spin.setValue(1.2)
        self.mag_locked = QCheckBox("Lock")
        self.mag_locked.setChecked(True)
        
        self.beam_size_mm_spin = QDoubleSpinBox()
        self.beam_size_mm_spin.setRange(50, 300)
        self.beam_size_mm_spin.setSingleStep(1)
        self.beam_size_mm_spin.setValue(154.8)
        self.beam_size_locked = QCheckBox("Lock")
        self.beam_size_locked.setChecked(True)

        self.pixel_size_um_spin = QDoubleSpinBox()
        self.pixel_size_um_spin.setRange(1.0, 50.0)
        self.pixel_size_um_spin.setSingleStep(0.5)
        self.pixel_size_um_spin.setValue(15)
        self.pixel_size_locked = QCheckBox("Lock")
        self.pixel_size_locked.setChecked(True)
        
        self.rel_element_spin = QDoubleSpinBox()
        self.rel_element_spin.setRange(1.0, 5.0)
        self.rel_element_spin.setSingleStep(0.1)
        self.rel_element_spin.setValue(2.8)
        self.rel_element_locked = QCheckBox("Lock")
        self.rel_element_locked.setChecked(True)
        
        self.f_cam_mm_spin = QDoubleSpinBox()
        self.f_cam_mm_spin.setRange(1, 1000)
        self.f_cam_mm_spin.setSingleStep(1)
        self.f_cam_mm_spin.setValue(455)
        self.f_cam_mm_locked = QCheckBox("Lock")
        self.f_cam_mm_locked.setChecked(True)
        
        self.nir_arm_checkbox = QCheckBox("NIR channel")
        
        advanced_layout.addWidget(QLabel("Grooves per mm"), 0, 0)
        advanced_layout.addWidget(self.grooves_mm_spin, 0, 1)
        advanced_layout.addWidget(self.grooves_mm_locked, 0, 2)
        
        advanced_layout.addWidget(self.resolution_label, 1, 0)
        advanced_layout.addWidget(self.resolution_slider, 1, 1)
        advanced_layout.addWidget(self.resolution_locked, 1, 2)

        advanced_layout.addWidget(QLabel("Magnification factor"), 2, 0)
        advanced_layout.addWidget(self.mag_factor_spin, 2, 1)
        advanced_layout.addWidget(self.mag_locked, 2, 2)
        
        advanced_layout.addWidget(QLabel("Beam diameter (mm)"), 3, 0)
        advanced_layout.addWidget(self.beam_size_mm_spin, 3, 1)
        advanced_layout.addWidget(self.beam_size_locked, 3, 2)

        advanced_layout.addWidget(QLabel("Pixel size (µm)"), 4, 0)
        advanced_layout.addWidget(self.pixel_size_um_spin, 4, 1)
        advanced_layout.addWidget(self.pixel_size_locked, 4, 2)
        
        advanced_layout.addWidget(QLabel("Resolution element (px)"), 5, 0)
        advanced_layout.addWidget(self.rel_element_spin, 5, 1)
        advanced_layout.addWidget(self.rel_element_locked, 5, 2)

        
        advanced_layout.addWidget(QLabel("Focal length camera (mm)"), 6, 0)
        advanced_layout.addWidget(self.f_cam_mm_spin, 6, 1)
        advanced_layout.addWidget(self.f_cam_mm_locked, 6, 2)
        
        advanced_layout.addWidget(self.nir_arm_checkbox, 7, 1)

        self.advanced_frame.setLayout(advanced_layout)
        layout.addWidget(self.advanced_frame)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_calculations)
        layout.addWidget(self.run_button)

        # Define tabs
        self.tabs = QTabWidget()
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.results_table.horizontalHeader().setStyleSheet(
            "QHeaderView::section { background-color: #4CAF50; color: white; font-weight: bold; }"
        )

        self.best_table = QTableWidget()
        self.best_table.setColumnCount(2)
        self.best_table.setHorizontalHeaderLabels(["Criteria", "Value"])
        self.best_table.horizontalHeader().setStyleSheet(
            "QHeaderView::section { background-color: #2196F3; color: white; font-weight: bold; }"
        )

        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)

        self.graph_canvas = FigureCanvas(plt.figure(figsize=(5, 4)))
        self.graph_images = []
        self.plot_index = 0
        self.prev_button = QPushButton("←")
        self.next_button = QPushButton("→")
        self.prev_button.clicked.connect(self.show_prev_plot)
        self.next_button.clicked.connect(self.show_next_plot)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)

        graph_tab_layout = QVBoxLayout()
        graph_tab_layout.addWidget(self.graph_canvas)
        graph_tab_layout.addLayout(nav_layout)

        graph_tab_widget = QWidget()
        graph_tab_widget.setLayout(graph_tab_layout)

        self.tabs.addTab(self.results_table, "Summary table")
        self.tabs.addTab(graph_tab_widget, "Plots")
        self.tabs.addTab(self.text_output, "Text details")
        self.tabs.addTab(self.best_table, "Best set-up")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def add_criteria_row(self):
        row = self.criteria_table.rowCount()
        self.criteria_table.insertRow(row)
        self.criteria_table.setItem(row, 0, QTableWidgetItem("New criteria"))
        
        benefit_cb = QCheckBox()
        benefit_cb.setChecked(True)
        
        cost_cb = QCheckBox()
        cost_cb.setChecked(False)
        
        benefit_cb.stateChanged.connect(lambda s, cb=cost_cb: (s == Qt.Checked) and cb.setChecked(False))
        cost_cb.stateChanged.connect(lambda s, cb=benefit_cb: (s == Qt.Checked) and cb.setChecked(False))
    
        self.criteria_table.setCellWidget(row, 1, benefit_cb)
        self.criteria_table.setCellWidget(row, 2, cost_cb)

    def del_criteria_row(self):
        row = self.criteria_table.currentRow()
        if row >= 0:
            self.criteria_table.removeRow(row)

    #def get_criteria_dict(self):
    #    criteria = {}
    #    for row in range(self.criteria_table.rowCount()):
    #        name_item = self.criteria_table.item(row, 0)
    #    return criteria
        
        
    def get_criteria_dict(self):
        criteria = {}
        for row in range(self.criteria_table.rowCount()):
            name_item = self.criteria_table.item(row, 0)
            benefit_widget = self.criteria_table.cellWidget(row, 1)
            cost_widget = self.criteria_table.cellWidget(row, 2)

            if benefit_widget.isChecked():
                crit_type = "benefit"
            elif cost_widget.isChecked():
                crit_type = "cost"
            else:
                crit_type = "none"

            criteria[name_item.text()] = {"type": crit_type}

        return criteria


        
    def toggle_advanced_params(self, checked):
        if checked:
            self.advanced_frame.show()
            self.advanced_button.setText("Hide advanced parameters")
        else:
            self.advanced_frame.hide()
            self.advanced_button.setText("Show advanced parameters")

    def update_module_diam_label(self):
        val = self.module_diam_slider.value()
        self.module_diam_label.setText(f"Module diameter (m): {val}")
        
    def update_resolution_label(self):
        val = self.resolution_slider.value()
        self.resolution_label.setText(f"Resolving power: {val}")
        
    def update_telescope_aperture_label(self):
        val = self.telescope_aperture_slider.value()
        self.telescope_aperture_label.setText(f"Telescope aperture (m): {val}")

    def show_plot_at_index(self, index):
        if 0 <= index < len(self.graph_images):
            self.graph_canvas.figure.clear()
            ax = self.graph_canvas.figure.add_subplot(111)
            img = plt.imread(self.graph_images[index])
            ax.imshow(img)
            ax.axis('off')
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
            
    def run_calculations(self):
        try:
            params = [
                float(self.module_diam_slider.value()),
                float(self.f_number_out_spin.value()),
                int(self.f_number_out_locked.isChecked()),
                float(self.d_core_out_spin.value()),
                int(self.d_core_out_locked.isChecked()),
                float(self.telescope_aperture_slider.value()),
                float(self.seeing_spin.value()),
                float(self.sky_aperture_spin.value()),
                int(self.sky_aperture_locked.isChecked()),
                self.tiptilt_checkbox.isChecked(),
                float(self.encircled_spin.value()),
                int(self.lambda_min_spin.value()),
                int(self.lambda_max_spin.value()),
                self.pseudoslit_checkbox.isChecked(),
                self.superpl_checkbox.isChecked(),
                float(self.grooves_mm_spin.value()),
                float(self.resolution_slider.value()),
                float(self.mag_factor_spin.value()),
                float(self.beam_size_mm_spin.value()),
                float(self.pixel_size_um_spin.value()),
                float(self.rel_element_spin.value()),
                float(self.f_cam_mm_spin.value()),
                int(self.f_cam_mm_locked.isChecked()),
                self.nir_arm_checkbox.isChecked()
            ]

            results_list = []

            results = marcot_hr_estimator(*params)
            self.results_table.setRowCount(len(results))
            
            #filter_data = {k: v for k, v in results.items() if isinstance(v, (int, float, np.ndarray)) and k != ""}
            

            def is_scalar(x):
                return isinstance(x, (int, float, np.integer, np.floating))

            # Know the length of the arrays
            lengths = []
            for k, v in results.items():
                if isinstance(v, np.ndarray):
                    lengths.append(v.size)
                elif isinstance(v, (list, tuple)):
                    lengths.append(len(v))
                    
            target_len = max(lengths) if lengths else 1

            cols = {}

            # Build the columna
            for k, v in results.items():
                if k == "" or v is None:
                    continue

                # Numpy arrays
                if isinstance(v, np.ndarray):
                    vec = np.asarray(v).ravel()
                    cols[k] = pd.Series(vec)

                # List
                elif isinstance(v, (list, tuple)):
                    cols[k] = pd.Series(list(v))

                # Scalar
                elif is_scalar(v):
                    cols[k] = pd.Series([float(v)] * target_len)


            df = pd.DataFrame(cols)
                
            df = df.reindex(range(target_len))

            df.to_csv("data/results_marcot.csv", sep="\t", index=False)
            print("\033[1;4;32mFile 'results_marcot.csv' was saved successfully\033[0m")

            # Colors for results_table
            for i, (key, value) in enumerate(results.items()):
                val_str = np.array2string(np.asarray(value), precision=3) if isinstance(value, (list, np.ndarray, np.generic)) else str(value)
                param_item = QTableWidgetItem(str(key))
                value_item = QTableWidgetItem(val_str)
                # Establecer texto en negro
                param_item.setForeground(QBrush(QColor(0, 0, 0)))
                value_item.setForeground(QBrush(QColor(0, 0, 0)))
                # Color de fondo alternado
                if i % 2 == 0:
                    param_item.setBackground(QBrush(QColor(240, 240, 240)))
                    value_item.setBackground(QBrush(QColor(240, 240, 240)))
                else:
                    param_item.setBackground(QBrush(QColor(255, 255, 255)))
                    value_item.setBackground(QBrush(QColor(255, 255, 255)))
                self.results_table.setItem(i, 0, param_item)
                self.results_table.setItem(i, 1, value_item)
            
            # Ajustar el tamaño de las columnas al contenido
            self.results_table.resizeColumnsToContents()

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                print_results(results)
            output_text = buffer.getvalue()


            snr_fraction = snr_cal("data/results_marcot.csv")
            snr_val = float(np.asarray(snr_fraction).ravel()[0])

            output_text += f"\nSNR Fraction: {snr_val:.3f}\n"
            
            # with contextlib.redirect_stdout(buffer):
            criteria = self.get_criteria_dict()
            multi_criteria("data/results_marcot.csv", criteria)
            output_text += "\nMultiple criterias calculated.\n"
            
            tables(criteria)

            self.text_output.setText(output_text)

            self.graph_images = [os.path.join("Figures", f) for f in os.listdir("Figures") if f.endswith(".png")]
            self.graph_images.sort()
            self.plot_index = 0
            if self.graph_images:
                self.show_plot_at_index(0)

            # Best set-up after MCM
            if os.path.exists("data/score_total_st_TOPSIS.csv"):
                df = pd.read_csv("data/score_total_st_TOPSIS.csv", sep='\t')
                list_columns = [col for col in df.columns if df[col].astype(str).str.startswith('[').any()]

                for col in list_columns:
                    df[col] = df[col].apply(lambda x: np.fromstring(x.strip('[] '), sep=' ') if isinstance(x, str) and '[' in x else x)

                criterios = [
                    "OTA diameter (m)", "Reduction cost factor", "Weight supported by the mount (kg)", "Selected commercial output core (microns)", "Expected efficiency",
                    "Resolution with commercial fibers", "SNR fraction","Number of OTA for high efficiency"
                ]

                for idx, row in df.iterrows():
                    score_total = row["score_total_st_TOPSIS"]
                    if isinstance(score_total, str) and score_total.startswith('['):
                        score_array = np.fromstring(score_total.strip('[] '), sep=' ')
                        best_index = np.argmax(score_array)
                    elif isinstance(score_total, (np.ndarray, list)):
                        best_index = np.argmax(score_total)
                    else:
                        best_index = 0
                    break

                self.best_table.setRowCount(0)
                row_counter = 0

                for criterio in criterios:
                    val = row[criterio]
                    if isinstance(val, str) and val.startswith('['):
                        val_array = np.fromstring(val.strip('[] '), sep=' ')
                        val_display = val_array[best_index] if len(val_array) > best_index else val_array[0]
                    elif isinstance(val, (np.ndarray, list)) and len(val) > best_index:
                        val_display = val[best_index]
                    elif isinstance(val, (np.ndarray, list)):
                        val_display = val[0]
                    else:
                        val_display = val

                    self.best_table.insertRow(row_counter)
                    criterio_item = QTableWidgetItem(str(criterio))
                    value_item = QTableWidgetItem(str(val_display))
                    # Establecer texto en negro
                    criterio_item.setForeground(QBrush(QColor(0, 0, 0)))
                    value_item.setForeground(QBrush(QColor(0, 0, 0)))
                    # Color de fondo alternado
                    if row_counter % 2 == 0:
                        criterio_item.setBackground(QBrush(QColor(225, 245, 255)))
                        value_item.setBackground(QBrush(QColor(225, 245, 255)))
                    else:
                        criterio_item.setBackground(QBrush(QColor(255, 255, 255)))
                        value_item.setBackground(QBrush(QColor(255, 255, 255)))
                    self.best_table.setItem(row_counter, 0, criterio_item)
                    self.best_table.setSortingEnabled(True)
                    self.best_table.setItem(row_counter, 1, value_item)
                    row_counter += 1

                self.best_table.insertRow(row_counter)
                
                self.best_table.setItem(row_counter, 0, QTableWidgetItem("score_total_st_TOPSIS"))
                if isinstance(score_total, str) and score_total.startswith('['):
                    score_array = np.fromstring(score_total.strip('[] '), sep=' ')
                    score_val = score_array[best_index]
                    
                    # Normalized score_total over 10
                    score_total = score_array - np.min(score_array)
                    #score_total_10 = score_total * (10 / np.max(score_total))
                    
                    # Save
                    #df['Score total over 10'] = None
                    #df.at[0, 'Score total over 10'] = score_total_10
                    #df.to_csv('data/score_total.csv', sep='\t', index=False)
                elif isinstance(score_total, (np.ndarray, list)):
                    score_val = score_total[best_index]
                    
                    # Normalized score_total over 10
                    #score_total = score_total - np.min(score_total)
                    #score_total_10 = score_total * (10 / np.max(score_total))
                    
                    # Save
                    #df['Score total over 10'] = None
                    #df.at[0, 'Score total over 10'] = score_total_10
                    #df.to_csv('data/score_total.csv', sep='\t', index=False)
                else:
                    score_val = score_total
                    score_total_10 = score_total

                score_item = QTableWidgetItem(str(score_val))
                #score_total_10_item = QTableWidgetItem(str(score_total_10))
                # Black text
                score_item.setForeground(QBrush(QColor(0, 0, 0)))
                #score_total_10_item.setForeground(QBrush(QColor(0, 0, 0)))
                #Background light orange
                score_item.setBackground(QBrush(QColor(255, 223, 186)))
                #score_total_10_item.setBackground(QBrush(QColor(255, 223, 186)))
                self.best_table.setItem(row_counter, 1, score_item)
                
                #self.best_table.insertRow(row_counter + 1)
                #self.best_table.setItem(row_counter + 1, 0, QTableWidgetItem("Score total over 10"))
                #self.best_table.setItem(row_counter + 1, 1, score_total_10_item)
            
            # Set the size of the cell to the text
            self.best_table.resizeColumnsToContents()

            # QMessageBox.information(self, "Success", "Calculation completed and results showed")

        except Exception as e:
            QMessageBox.critical(self, "Error", traceback.format_exc())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MarcotApp()
    window.resize(900, 1000)
    window.show()
    sys.exit(app.exec_())
