import os
import pickle
import pandas as pd

print("🔍 VERIFICACIÓN RÁPIDA DEL MODELO")
print("=" * 50)

MODELS_DIR = r"C:\Users\Mario Leyser\PROYECTO_BASE\Interboot\INTERBOOT\models"

# Verificar archivos
files = os.listdir(MODELS_DIR)
print("Archivos en models/:")
for file in files:
    print(f"  📄 {file}")

# Intentar cargar modelo
try:
    model_path = os.path.join(MODELS_DIR, "best_sales_model.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✅ Modelo cargado correctamente")
    print(f"   Tipo: {type(model)}")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")

# Intentar cargar feature_info
try:
    feature_path = os.path.join(MODELS_DIR, "feature_info.pkl")
    with open(feature_path, 'rb') as f:
        features = pickle.load(f)
    print("✅ Feature info cargado correctamente")
    print(f"   Keys: {list(features.keys())}")
except Exception as e:
    print(f"❌ Error cargando features: {e}")

print("=" * 50)