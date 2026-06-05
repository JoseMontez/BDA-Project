from pathlib import Path

def is_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# PROJECT ROOT
if is_colab():
    PROJECT_ROOT = Path("/content/drive/MyDrive/e_redes_v2/e_redes")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent



# BASE DIRECTORIES
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
STREAM_DIR = DATA_DIR / "stream"
PLOT_DIR = PROJECT_ROOT / "plots"


# RAW DATA (Notebook 1)
DATA_EREDES_DIR = DATA_DIR / "raw" / "e_redes"
DATA_CP_DIR = DATA_DIR / "raw" / "cp"
DATA_MUN_DIR = DATA_DIR / "raw" / "municipalities"
DATA_PROCESSED = DATA_DIR / "processed"

EREDES_CSV = str(DATA_EREDES_DIR / "consumos_horario_codigo_postal.csv")
POSTAL_SHP = str(DATA_CP_DIR / "CP4_EstimativaPoligonos.shp")
MUNICIPALITY_JSON = str(DATA_MUN_DIR / "georef-portugal-concelho-millesime.shp")

WEIGHTS_PARQUET = str(DATA_PROCESSED / "postal_municipality_weights.parquet")
OUTPUT_PARQUET = str(DATA_PROCESSED / "eredes_clean.parquet")


# FEATURE ENGINEERING (Notebook 2)  
DATA_MODEL = DATA_DIR / "model_input"


EREDES_PARQUET = str(DATA_PROCESSED / "eredes_clean.parquet")
MODEL_INPUT_PARQUET = str(DATA_MODEL / "model_input.parquet")
TARGET_COL = "total_active_energy_kwh"


# STREAMING / INFERENCE (Notebook 4)
STREAM_INPUT_DIR = STREAM_DIR / "input"
STREAM_OUTPUT_DIR = STREAM_DIR / "output"
CHECKPOINT_DIR = STREAM_DIR / "checkpoints"

MODEL_LR_PIPELINE_DIR = str(MODEL_DIR) + '/lr_pipeline'
STREAM_SPLIT_DATE = "2023-09-16"