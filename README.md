# Voice Authentication CNN (Optimized)
A voice authentication system using pre-trained CNN embeddings with both CLI and Streamlit UI.

## Setup
```bash
python -m pip install -r requirements.txt
```

## Run UI
```bash
streamlit run app.py
```
Use the **Enroll** tab to enter `name` and `register number`, then either upload audio or use live recording.

## CLI Usage

### Enroll single user
```bash
python voice_auth.py -t enroll -n "Alice" -f data/wav/Amy/700-122866-0000.flac
```

### Enroll from CSV
CSV should contain `filename,speaker` columns.
```bash
python voice_auth.py -t enroll -f path/to/list.csv
```

### Recognize user
```bash
python voice_auth.py -t recognize -f data/wav/Amy/700-122866-0000.flac
```

## Notes
- Uses TensorFlow SavedModel-compatible loading for modern Keras/TensorFlow stacks.
- Enrolled embeddings are stored in `data/embed`.
- Recognition is based on `COST_METRIC` and `THRESHOLD` from `parameters.py`.
