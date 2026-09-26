# ==========================================
# CONFIGURATION & MAPPING
# ==========================================

LABEL_MAPPING = {
    'Benign': 0,
    'FTP-BruteForce': 1, 'SSH-Bruteforce': 1, 'Brute Force -Web': 1, 'Brute Force -XSS': 1,
    'DoS attacks-GoldenEye': 2, 'DoS attacks-Hulk': 2, 'DoS attacks-SlowHTTPTest': 2, 'DoS attacks-Slowloris': 2,
    'DDOS attack-HOIC': 3, 'DDoS attacks-LOIC-HTTP': 3, 'DDOS attack-LOIC-UDP': 3,
    'Bot': 4, 'Infilteration': 4, 'SQL Injection': 4
}

CLASS_NAMES = {
    0: "Class 0", 1: "Class 1", 2: "Class 2", 3: "Class 3", 4: "Class 4"
}

# Default hyperparameters
DEFAULT_HIDDEN_DIM = 64
DEFAULT_SEQ_LEN = 5
DEFAULT_LR = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_NUM_CLASSES = 5