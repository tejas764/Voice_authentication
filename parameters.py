# Signal processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512
BUCKET_STEP = 1
MAX_SEC = 10

# Model
MODEL_FILE = "voice_auth_model_cnn"
INPUT_SHAPE = (NUM_FFT, None, 1)

# IO
EMBED_LIST_FILE = "data/embed"

# Recognition
COST_METRIC = "cosine"  # "cosine" or "euclidean"
THRESHOLD = 0.35
