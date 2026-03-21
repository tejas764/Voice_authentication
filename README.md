# Voice Authentication (CNN + Live Recording UI)

Voice authentication system using pre-trained CNN embeddings, optimized for modern TensorFlow/Keras and packaged with a user-friendly Streamlit interface.

## Features
- Voice enrollment using `Name + Register Number`
- Live microphone recording in UI (no manual file conversion required)
- Voice recognition with similarity score and threshold decision
- Bulk enrollment from CSV
- Optimized recognition path using in-memory embedding cache and vectorized distance scoring

## Project Structure
- `app.py` - Streamlit UI
- `voice_auth.py` - CLI entry point
- `auth_service.py` - Enrollment/recognition service logic
- `model_utils.py` - TensorFlow/Keras SavedModel compatibility loader
- `feature_extraction.py`, `preprocess.py` - audio preprocessing + embedding extraction
- `data/embed/` - saved embeddings and user metadata (`users.csv`)

## Requirements
- Python 3.10+
- Windows/macOS/Linux

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Run Web UI
```bash
python -m streamlit run app.py
```

### UI Workflow
1. Open **Enroll** tab
2. Enter **Student name** and **Register number**
3. Upload `.wav/.flac` OR use **live recording**
4. Click **Enroll User**
5. Open **Recognize** tab and upload/record unknown voice to verify identity

## CLI Usage
### Enroll single user
```bash
python voice_auth.py -t enroll -n "Alice" -f data/wav/Amy/700-122866-0000.flac
```

### Enroll from CSV
CSV should include columns: `filename,speaker`
```bash
python voice_auth.py -t enroll -f path/to/list.csv
```

### Recognize user
```bash
python voice_auth.py -t recognize -f data/wav/Amy/700-122866-0000.flac
```

## Configuration
Tune in `parameters.py`:
- `THRESHOLD` - decision boundary for recognition acceptance
- `COST_METRIC` - `cosine` or `euclidean`
- `MAX_SEC` - max audio duration processed

## Notes
- Model files are loaded from `voice_auth_model_cnn/`.
- Enrolled vectors are stored as `.npy` files in `data/embed/`.
- User metadata for UI identity display is stored in `data/embed/users.csv`.
