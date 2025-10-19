# 1️⃣ Cargar datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ENTRENAMIENTO DEL MODELO DE PREDICCIÓN DE VENTAS")
print("=" * 60)

# Configurar rutas
base_dir = r"C:\Users\Mario Leyser\PROYECTO_BASE\Interboot\INTERBOOT"
processed_path = os.path.join(base_dir, "data", "processed")
models_path = os.path.join(base_dir, "models")
os.makedirs(models_path, exist_ok=True)

# Cargar datos procesados
try:
    clean_data_path = os.path.join(processed_path, 'sales_data_cleaned.csv')
    df = pd.read_csv(clean_data_path, parse_dates=['date'])
    print("✓ Datos limpios cargados exitosamente")
except FileNotFoundError:
    print("Cargando datos originales...")
    raw_path = os.path.join(base_dir, "data", "raw", "train.csv")
    df = pd.read_csv(raw_path, parse_dates=['date'])

print(f"Dataset shape: {df.shape}")

# 2️⃣ Seleccionar características
print("\n" + "=" * 60)
print("SELECCIÓN DE CARACTERÍSTICAS")
print("=" * 60)

# Crear características temporales
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['weekofyear'] = df['date'].dt.isocalendar().week
df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

# Codificar variables categóricas
family_dummies = pd.get_dummies(df['family'], prefix='family')
store_dummies = pd.get_dummies(df['store_nbr'], prefix='store')

# Combinar características
features = pd.concat([
    pd.DataFrame({
        'year': df['year'],
        'month': df['month'],
        'day': df['day'],
        'dayofweek': df['dayofweek'],
        'quarter': df['quarter'],
        'weekofyear': df['weekofyear'],
        'is_weekend': df['is_weekend'],
        'onpromotion': df['onpromotion']
    }),
    family_dummies,
    store_dummies
], axis=1)

target = df['sales']

print("Características seleccionadas:")
print(f"- Número de características: {features.shape[1]}")
print(f"- Variables temporales: 8")
print(f"- Variables de familia: {family_dummies.shape[1]}")
print(f"- Variables de tienda: {store_dummies.shape[1]}")
print(f"- Variable objetivo: ventas (sales)")

# 3️⃣ Separar datos
print("\n" + "=" * 60)
print("DIVISIÓN DE DATOS")
print("=" * 60)

# Dividir por fecha (más realista para series temporales)
split_date = df['date'].quantile(0.8)  # 80% train, 20% test
train_mask = df['date'] < split_date
test_mask = df['date'] >= split_date

X_train = features[train_mask]
X_test = features[test_mask]
y_train = target[train_mask]
y_test = target[test_mask]

print(f"División temporal:")
print(f"- Fecha de división: {split_date}")
print(f"- Entrenamiento: {len(X_train):,} muestras ({len(X_train)/len(features)*100:.1f}%)")
print(f"- Prueba: {len(X_test):,} muestras ({len(X_test)/len(features)*100:.1f}%)")
print(f"- Período entrenamiento: {df[train_mask]['date'].min()} a {df[train_mask]['date'].max()}")
print(f"- Período prueba: {df[test_mask]['date'].min()} a {df[test_mask]['date'].max()}")

# 4️⃣ Entrenar modelo
print("\n" + "=" * 60)
print("ENTRENAMIENTO DE MODELOS")
print("=" * 60)

# Modelo 1: Random Forest
print("Entrenando Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("✓ Random Forest entrenado")

# Modelo 2: Linear Regression
print("Entrenando Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("✓ Linear Regression entrenado")

# 5️⃣ Evaluar modelo
print("\n" + "=" * 60)
print("EVALUACIÓN DE MODELOS")
print("=" * 60)

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluar modelo y retornar métricas"""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Métricas adicionales
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n{model_name}:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {
        'model': model,
        'predictions': y_pred,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

# Evaluar ambos modelos
rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
lr_results = evaluate_model(lr_model, X_test, y_test, "Linear Regression")

# Seleccionar el mejor modelo
if rf_results['rmse'] < lr_results['rmse']:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_results = rf_results
else:
    best_model = lr_model
    best_model_name = "Linear Regression"
    best_results = lr_results

print(f"\n🏆 MEJOR MODELO: {best_model_name}")
print(f"  RMSE: ${best_results['rmse']:.2f}")
print(f"  R²: {best_results['r2']:.4f}")

# Visualizar resultados
plt.figure(figsize=(15, 5))

# Gráfico 1: Predicciones vs Real
plt.subplot(1, 3, 1)
plt.scatter(y_test, best_results['predictions'], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Ventas Reales')
plt.ylabel('Ventas Predichas')
plt.title(f'Predicciones vs Real - {best_model_name}')

# Gráfico 2: Distribución de errores
plt.subplot(1, 3, 2)
errors = y_test - best_results['predictions']
plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(errors.mean(), color='red', linestyle='--', label=f'Media: {errors.mean():.2f}')
plt.xlabel('Error de Predicción')
plt.ylabel('Frecuencia')
plt.title('Distribución de Errores')
plt.legend()

# Gráfico 3: Comparación de modelos
plt.subplot(1, 3, 3)
models = ['Random Forest', 'Linear Regression']
rmse_values = [rf_results['rmse'], lr_results['rmse']]
colors = ['green' if x == min(rmse_values) else 'blue' for x in rmse_values]
bars = plt.bar(models, rmse_values, color=colors, alpha=0.7)
plt.ylabel('RMSE ($)')
plt.title('Comparación de RMSE entre Modelos')
plt.xticks(rotation=45)

# Añadir valores en las barras
for bar, value in zip(bars, rmse_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'${value:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Guardar modelo y preprocesadores
print("\n" + "=" * 60)
print("GUARDANDO MODELO Y RECURSOS")
print("=" * 60)

# Guardar el mejor modelo
model_path = os.path.join(models_path, 'best_sales_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"✓ Modelo guardado: {model_path}")

# Guardar información de características
feature_info = {
    'feature_names': features.columns.tolist(),
    'family_categories': df['family'].unique().tolist(),
    'store_categories': df['store_nbr'].unique().tolist(),
    'min_date': df['date'].min(),
    'max_date': df['date'].max(),
    'model_name': best_model_name,
    'performance': best_results
}

feature_info_path = os.path.join(models_path, 'feature_info.pkl')
with open(feature_info_path, 'wb') as f:
    pickle.dump(feature_info, f)
print(f"✓ Información de características guardada: {feature_info_path}")

# Guardar métricas del modelo
metrics_df = pd.DataFrame({
    'Modelo': ['Random Forest', 'Linear Regression'],
    'MAE': [rf_results['mae'], lr_results['mae']],
    'RMSE': [rf_results['rmse'], lr_results['rmse']],
    'R2': [rf_results['r2'], lr_results['r2']],
    'MAPE': [rf_results['mape'], lr_results['mape']]
})

metrics_path = os.path.join(models_path, 'model_metrics.csv')
metrics_df.to_csv(metrics_path, index=False)
print(f"✓ Métricas guardadas: {metrics_path}")

print("\n" + "=" * 60)
print("RESUMEN DEL ENTRENAMIENTO")
print("=" * 60)
print(f"✅ Modelo entrenado: {best_model_name}")
print(f"✅ Rendimiento (RMSE): ${best_results['rmse']:.2f}")
print(f"✅ Exactitud (R²): {best_results['r2']:.4f}")
print(f"✅ Error porcentual (MAPE): {best_results['mape']:.2f}%")
print(f"✅ Archivos guardados en: {models_path}")
print(f"✅ El modelo está listo para deployment")