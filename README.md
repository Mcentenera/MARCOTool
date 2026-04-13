# MARCOTool

**MARCOTool** (Modular ARray COst Tool) is a GUI application for estimating and optimising the design of modular telescope arrays coupled to photonic lanterns and high-resolution spectrographs.

Given a set of optical and detector parameters, MARCOTool computes:

- The number and geometry of commercial off-the-shelf OTAs that fit a target module diameter.
- Input/output fiber selection and photonic lantern efficiency.
- Spectrograph sizing (beam diameter, volume, mass, cost).
- Spectral resolution with commercial fibers.
- Signal-to-Noise Ratio (SNR) fraction.
- Multi-criteria ranking of configurations using fuzzy TOPSIS and MABAC with three weight-determination methods (Statistical Variance, CRITIC, MEREC).

---

## Project structure

```
MARCOTool/
├── main.py                  # Application entry point (MARCOTool.py)
├── src/
│   ├── __init__.py
│   ├── estimator.py         # Core optical/photonic/spectrograph estimator
│   └── utils.py             # Results printing, SNR, multi-criteria analysis
├── data/
│   ├── Commercial_OTA.txt   # Commercial OTA database
│   ├── Commercial_fibers.txt# Commercial fiber database
│   └── ...                  # Other input data files
├── Images/
│   └── logo.png
├── Figures/                 # Output plots (generated at runtime)
├── requirements.txt
└── README.md
```

---

## Installation

**Python 3.9 or higher** is recommended.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/MARCOTool.git
   cd MARCOTool
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   .venv\Scripts\activate         # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

```bash
python MARCOTool.py
```

The GUI will open. Set the telescope, fiber, and spectrograph parameters in the tabs, then click **Run**.

Results are saved to `data/results_marcot.csv`. Figures are saved to `Figures/`.

---

## Input parameters

| Tab | Parameter | Description |
|-----|-----------|-------------|
| Basic | Module diameter (m) | Target diameter of the telescope module |
| Basic | F/# output | Output focal ratio of the photonic lantern |
| Basic | Seeing FWHM (arcsec) | Atmospheric seeing |
| Basic | Sky aperture (arcsec) | On-sky aperture for fiber injection |
| Basic | Wavelength range (nm) | Minimum and maximum wavelengths |
| Detector | Plate scale (um/arcsec) | Detector plate scale |
| Detector | Read noise (e⁻) | CCD/detector read noise |
| Detector | Quantum efficiency | Detector QE |
| Spectrograph | Grooves/mm | Grating groove density |
| Spectrograph | Resolving power | Target spectral resolution R = λ/Δλ |
| Spectrograph | Beam diameter (mm) | Collimated beam size at the grating |
| Criteria | — | Define benefit/cost criteria for multi-criteria ranking |

---

## Output tabs

- **Summary table** — all computed parameters for each commercial OTA alternative.
- **Plots** — efficiency vs. NA, and other diagnostic figures.
- **Text details** — formatted numerical summary and SNR fraction.
- **Best set-up** — highest-ranked configuration according to fuzzy TOPSIS.

---

## Dependencies

See `requirements.txt`. Main libraries:

- [PyQt5](https://pypi.org/project/PyQt5/) — GUI framework
- [NumPy](https://numpy.org/) — numerical computation
- [pandas](https://pandas.pydata.org/) — tabular data handling
- [Matplotlib](https://matplotlib.org/) — plotting
- [SciPy](https://scipy.org/) — statistical functions
- [seaborn](https://seaborn.pydata.org/) — statistical visualisation

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
