# craquelure-fusion

Does adding a height map stop a painted stroke being mistaken for a crack?

Synthetic data only (no real scanner data yet). See `notes/` for the study notes.

## Pipeline

1. **Generate** (`craquelure_fusion/generate.py`): one fake painting with a known crack and a painted stroke (the confuser).
2. **Fixed evaluation set** (`craquelure_fusion/eval_set.py`): 30 samples, seeds 1000 to 1029, saved in `data/eval/` with a checksum manifest.
3. **Models** (`craquelure_fusion/baseline.py`): Model A (colour only) and Model B (colour + height). Classical, no learning yet.
4. **Evaluate** (`craquelure_fusion/evaluate.py`, `metrics.py`): tolerant precision and recall, and the stroke false-alarm rate.
5. **Show** (`app/streamlit_app.py`): the pipeline diagram, one sample, and the results.

## Run it

    pip install -r requirements.txt
    python -m pytest -q                          # tests
    python -m craquelure_fusion.eval_set --verify --out data/eval
    python -m craquelure_fusion.evaluate         # writes results/results.csv
    streamlit run app/streamlit_app.py           # the presentation

To rebuild the evaluation set from scratch: `python -m craquelure_fusion.eval_set --out data/eval`.
Do not rebuild it casually: a rebuild with changed settings makes results incomparable.

`practice/` holds exercises and is not part of the test run.
