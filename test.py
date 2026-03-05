import scipy.io
from pathlib import Path

BASE_DIR = Path(__file__).parent
data_folder = BASE_DIR / "DATA FOLDER"
file_path = data_folder / "ae2224I_measurement_data_subj1_C1.mat"

mat = scipy.io.loadmat(str(file_path))
print(mat)
