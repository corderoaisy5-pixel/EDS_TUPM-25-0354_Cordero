EDS_TUPM-25-0354_Cordero
MAN-01: CNC Surface Roughness Analytics
Course: Computer Programming
Academic Year: 2026
Student: Aisy Jibril B. Cordero
Student ID: TUPM-25-0354
Pillar: Pillar 4 — Manufacturing & Production
Topic: MAN-01 — CNC Surface Roughness Analytics

Project Description
This project implements a production-grade Python data analytics pipeline for analyzing CNC (Computer Numerical Control) turning surface roughness data. The pipeline ingests real-world machining experiment data from Kaggle, performs automated cleaning, computes engineering-grade statistical metrics using NumPy, and generates both static and animated visualizations. Results are reported in IEEE two-column research format.
The dataset is sourced from CNC Turning: Roughness, Forces and Tool Wear by A. D. Rigueto et al. (adorigueto/cnc-turning-roughness-forces-and-tool-wear on Kaggle), comprising three experimental CSV files (Exp1, Exp2, Prep) totaling 824 records across 27 machining variables including surface roughness parameters (Ra, Rz, Rt), cutting forces (Fx, Fy, Fz), and process parameters (feed rate, cutting speed, depth of cut).

Repository Structure
EDS_TUPM-25-0354_Cordero/
│
├── main.py                    # Main Python pipeline (OOP, modular)
├── requirements.txt           # Required Python libraries
├── README.md                  # This file
│
├── data/
│   ├── Exp1.csv               # Kaggle source file 1 (324 rows)
│   ├── Exp2.csv               # Kaggle source file 2 (288 rows)
│   ├── Prep.csv               # Kaggle source file 3 (212 rows)
│   ├── dataset_original.csv   # Auto-generated merged dataset (824 rows)
│   └── dataset_cleaned.csv    # Auto-generated cleaned & filtered dataset (572 rows)
│
└── outputs/
    ├── plot1_histogram.png              # Fig. 1 — Ra distribution with KDE
    ├── plot2_boxplot_speed.png          # Fig. 2 — Ra: Low vs High cutting speed
    ├── plot3_heatmap.png                # Fig. 3 — Pearson correlation heatmap
    ├── plot4_scatter_feed_vs_Ra.png     # Fig. 4 — Feed rate vs Ra scatter
    ├── plot5_outlier_detection.png      # Fig. 5 — Z-score outlier detection
    ├── anim1_rolling_mean_Ra.gif        # Fig. 6 — Animated rolling mean of Ra
    └── anim2_progressive_histogram.gif  # Fig. 7 — Progressive histogram build-up

Pipeline Architecture
[Data Ingestion] → [Data Cleaning] → [Analytics] → [Visualization] → [Report]
   Exp1.csv            Duplicates       Mean           Histogram        Console
   Exp2.csv            Null Fill        Median         Boxplot          Summary
   Prep.csv            Type Fix         Std/Var        Heatmap
                       Unique Filter    Skewness       Scatter
                       (f ≤ 50th pct)   IQR/Outliers   Outlier Plot
                                        Correlation    Rolling Mean GIF
                                        t-test         Histogram GIF

How to Run
1. Clone the Repository
bashgit clone https://github.com/corderoaisy5-pixel/EDS_TUPM-25-0354_Cordero.git
cd EDS_TUPM-25-0354_Cordero
2. Set Up Virtual Environment (Recommended)
bashpython -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
3. Install Dependencies
bashpip install -r requirements.txt
4. Add Dataset Files
Place the following Kaggle CSV files inside the data/ folder:

Exp1.csv
Exp2.csv
Prep.csv

Download from: Kaggle — CNC Turning: Roughness, Forces and Tool Wear
5. Run the Pipeline
bashpython main.py
The pipeline will automatically:

Merge Exp1.csv, Exp2.csv, and Prep.csv into data/dataset_original.csv
Clean the data and apply a unique feed rate filter (f ≤ 50th percentile)
Save the cleaned data to data/dataset_cleaned.csv
Compute all statistical metrics and print an engineering summary
Save 5 static plots and 2 animated GIFs to outputs/


Unique Data Filter Logic
To ensure a mathematically unique data slice per the project requirement:

Filter: Records where feed rate f ≤ 0.10 mm/rev (50th percentile) are retained.

This isolates fine-finish machining conditions — the lower half of feed rates where surface roughness behavior is most critical for precision manufacturing analysis.
Result: 824 raw records → 572 filtered records used for analysis.

Key Results
MetricValueMean Ra0.5500 µmMedian Ra0.5390 µmStd Deviation0.1885 µmVariance0.0355 µm²Skewness0.3987 (right-skewed)IQR0.2823 µmStrongest CorrelateRz (r = +0.9088)Cutting Speed EffectNot significant (p = 0.1516)

Dependencies
LibraryPurposenumpyStatistical computations (mean, std, variance, IQR)pandasData ingestion, cleaning, and filteringmatplotlibStatic plots and animated GIFsseabornHeatmap and aesthetic stylingscipySkewness, KDE, t-test, z-score, linear regressionpillowGIF animation writer
Install all with:
bashpip install -r requirements.txt

Dataset Citation
A. D. Canal, A. V. Borille, "CNC Turning: Roughness, Forces and Tool Wear Dataset," Kaggle, 2022.
Available: https://www.kaggle.com/datasets/adorigueto/cnc-turning-roughness-forces-and-tool-wear

License
For academic use only — Technological University of the Philippines, Computer Programming Course, 2026.
