Reproducibility Steps:

1. Create environment:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Train baseline:
   python train_baseline.py

3. Train geometry-aware:
   python train_geom_fp1.py

4. Evaluate:
   python eval_final_report.py

5. Figures:
   python make_ch5_figures_B.py
   python make_ch5_figures_C.py
