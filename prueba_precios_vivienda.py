# Estructurando el código para convertirlo en una aplicación FastAPI

fastapi_code = """
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import prince
from scipy import stats
from io import StringIO

app = FastAPI()

# Variable global para almacenar los datos
df = None

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    global df
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    
    return {"filename": file.filename, "columns": df.columns.tolist()}

@app.get("/data-summary/")
def data_summary():
    global df
    if df is None:
        raise HTTPException(status_code=400, detail="No data loaded.")
    
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.apply(str).to_dict(),
        "columns": df.columns.tolist(),
        "description": df.describe().to_dict(),
    }
    return summary

@app.get("/clean-data/")
def clean_data():
    global df
    if df is None:
        raise HTTPException(status_code=400, detail="No data loaded.")
    
    # Ejemplo de una limpieza de datos básica
    # Puedes ajustar esto según las necesidades específicas de tu análisis
    df = df.dropna()  # Elimina filas con valores NaN
    df = df[df['precio'] > 0]  # Filtra por ejemplo solo precios positivos

    return {"message": "Data cleaned", "new_shape": df.shape}

@app.get("/correlation/")
def correlation_matrix():
    global df
    if df is None:
        raise HTTPException(status_code=400, detail="No data loaded.")
    
    correlation_matrix = df.corr().to_dict()
    
    return {"correlation_matrix": correlation_matrix}

@app.get("/pca/")
def perform_pca():
    global df
    if df is None:
        raise HTTPException(status_code=400, detail="No data loaded.")
    
    pca = prince.PCA(df, n_components=2, n_iter=3, rescale_with_mean=True, rescale_with_std=True)
    pca_result = pca.fit_transform(df.select_dtypes(include=[np.number]))
    
    return pca_result.to_dict()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

# Guardar este código en un archivo Python para FastAPI
output_path = '/mnt/data/fastapi_app.py'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(fastapi_code)

output_path
