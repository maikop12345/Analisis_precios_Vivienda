# -*- coding: utf-8 -*-
"""Prueba - Precios vivienda .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RoX-Q7vnZim-mJYeCQ4Cg0ypbABC1Whx

**Importar Liberias**
"""

!pip install pandas prince matplotlib
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn  as sns
import time
import prince
from scipy import stats

"""**Lectura de Datos**"""

tpv = pd.read_csv('/content/drive/MyDrive/Prueba/train_precios_vivienda.csv',encoding='utf-8')

"""Revisón Cantidad de filas y columnas"""

tpv.shape

tpv.head()

"""Se realiza una primera visualización de la Data donde se identicia que esta leyendo algunas letras como caracteres especiales como caracteres especiales

**Verificación Tipo de Datos**
"""

tpv.dtypes

"""**Se realiza un visualización de los nombres de las columnas**"""

tpv.columns

tpv.describe()

tpv.info()

"""**Preparación y Limpieza marco de Datos**"""

# Diccionario de reemplazos

tpv = tpv.apply(lambda x: x.str.replace('Ã©', 'e') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Ã¡', 'a') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Ã­', 'i') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Ã³', 'o') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Ãº', 'u') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Ã‰', 'E') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Ã\u0081', 'A') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Ã\u008D', 'I') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Ã“', 'O') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Ãš', 'U') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Ã±', 'n') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Ã‘', 'N') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Â¿', '¿') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Â¡', '¡') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Â°', '°') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Ã¼', 'u') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Ãœ', 'U') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Ã ', 'a') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('Ã€', 'A') if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('í', 'I'  ) if x.dtype == "object" else x)
tpv = tpv.apply(lambda x: x.str.replace('É', 'E'  ) if x.dtype == "object" else x)

#Realizamos una nueva visualización
tpv.head()

#Revisión de NA
missing_values = tpv.isna().sum()
missing_values = missing_values[missing_values != 0]
print(missing_values)

"""
```
1,fecha_aprobación: La fecha no sera tomada para este analisis
2,tipo_subsidio: debido a la gran cantidad de datos
3,barrio : Se omite el dato
4,sector : Se omite el dato
5,"descripcion_tipo_inmueble : Se realiza uan revisión mas detallada y se verifica que solo hay 2767 datos registrados, lo que ocasiona un faltante de 8804 datos, se omite esta columna."
6,"descripcion_uso_inmueble : Se realiza uan revisión mas detallada y se verifica que solo hay 2790 datos registrados, lo que ocasiona un faltante de 8791 datos, se omite esta columna."
7,"descripcion_clase_inmueble : Se realiza uan revisión mas detallada y se verifica que solo hay 8833 datos registrados, lo que ocasiona un faltante de 8804 datos, se omite esta columna."
8,accesorios 3379
9,area_actividad : Se omite el dato
10,observaciones_estructura : Se omite el dato
11,comedor: se valida el tipo de variable y se transmorfa
12,bano_privado: se valida el tipo de variable y se transmorfa
13,observaciones_dependencias: Se omite el dato
14,"numero_garaje_1: se omite esta columna, se toma el total de garajes"
15,"numero_garaje_2: se omite esta columna, se toma el total de garajes"
16,"garaje_cubierto_2: se omite esta columna, se toma el total de garajes"
17,"numero_garaje_4: se omite esta columna, se toma el total de garajes"
18,"matricula_garaje_5: se omite esta columna, se toma el total de garajes"
19,"garaje_cubierto_5: se omite esta columna, se toma el total de garajes"
20,"garaje_doble_5: se omite esta columna, se toma el total de garajes"
21,"garaje_paralelo_5: se omite esta columna, se toma el total de garajes"
22,"garaje_servidumbre_5: se omite esta columna, se toma el total de garajes"
23,"numero_deposito_3: se omite esta columna, numero_total_depositos"
24,"matricula_inmobiliaria_deposito_3: se omite esta columna, numero_total_depositos"
25,"numero_deposito_4: se omite esta columna, numero_total_depositos"
26,"matricula_inmobiliaria_deposito_4: se omite esta columna, numero_total_depositos"
27,observaciones_generales_construccion 15: se omite los datos
28,Latitud: se imite el dato
```
"""

#omitimos NA
tpv = tpv.dropna(subset=['barrio', 'sector','area_actividad','observaciones_estructura','observaciones_dependencias','Latitud'])

missing_values = tpv.isna().sum()
missing_values = missing_values[missing_values != 0]
print(missing_values)

"""Se omiten las columanas con descripciónes"""

# Lista de las columnas que deseas mantener
columnas_a_mantener = ['cod',
    'objeto', 'motivo', 'proposito', 'tipo_avaluo', 'tipo_credito', 'departamento_inmueble',
    'municipio_inmueble', 'barrio', 'sector', 'direccion_inmueble_informe',
    'alcantarillado_en_el_sector', 'acueducto_en_el_sector', 'gas_en_el_sector',
    'energia_en_el_sector', 'telefono_en_el_sector', 'vias_pavimentadas',
    'sardineles_en_las_vias', 'andenes_en_las_vias', 'estrato', 'barrio_legal',
    'topografia_sector', 'condiciones_salubridad', 'transporte', 'demanda_interes',
    'paradero', 'alumbrado', 'arborizacion', 'alamedas', 'ciclo_rutas',
    'nivel_equipamiento_comercial', 'alcantarillado_en_el_predio',
    'acueducto_en_el_predio', 'gas_en_el_predio', 'energia_en_el_predio',
    'telefono_en_el_predio', 'tipo_inmueble', 'uso_actual', 'clase_inmueble',
    'ocupante', 'sometido_a_propiedad_horizontal', 'altura_permitida',
    'observaciones_altura_permitida', 'aislamiento_posterior',
    'observaciones_aislamiento_posterior', 'aislamiento_lateral',
    'observaciones_aislamiento_lateral', 'antejardin', 'observaciones_antejardin',
    'indice_ocupacion', 'indice_construccion',
    'observaciones_indice_construccion', 'predio_subdividido_fisicamente',
    'unidades', 'contadores_agua', 'contadores_luz', 'accesorios', 'area_valorada',
    'condicion_ph', 'numero_piso', 'numero_de_edificios',
    'rph', 'porteria', 'citofono', 'bicicletero', 'piscina',
    'tanque_de_agua', 'club_house', 'garaje_visitantes', 'teatrino', 'sauna',
    'vigilancia_privada', 'tipo_vigilancia', 'administracion', 'vetustez',
    'pisos_bodega', 'estructura', 'ajustes_sismoresistentes', 'cubierta', 'fachada',
    'tipo_fachada', 'estructura_reforzada', 'danos_previos', 'material_de_construccion',
    'iluminacion', 'ventilacion', 'irregularidad_planta',
    'irregularidad_altura', 'habitaciones',
    'estar_habitacion', 'cuarto_servicio', 'closet', 'sala', 'comedor', 'bano_privado',
    'bano_social', 'bano_servicio', 'cocina', 'estudio', 'balcon', 'terraza',
    'patio_interior', 'jardin', 'zona_de_ropas', 'zona_verde_privada', 'local',
    'oficina', 'bodega', 'calidad_acabados_pisos', 'estado_acabados_muros',
    'calidad_acabados_muros', 'estado_acabados_techos', 'calidad_acabados_techos',
    'estado_acabados_madera', 'calidad_acabados_madera', 'estado_acabados_metal',
    'calidad_acabados_metal', 'estado_acabados_banos', 'calidad_acabados_banos',
    'estado_acabados_cocina', 'calidad_acabados_cocina', 'tipo_garaje',
    'numero_total_de_garajes', 'total_cupos_parquedaro', 'tipo_deposito',
    'numero_total_depositos', 'area_privada', 'valor_area_privada', 'area_garaje',
    'valor_area_garaje', 'area_deposito', 'valor_area_deposito', 'area_terreno',
    'valor_area_terreno', 'area_construccion', 'valor_area_construccion', 'area_otros',
    'valor_area_otros', 'area_libre', 'valor_area_libre', 'valor_total_avaluo', 'Longitud', 'Latitud'
]

# Filtrar el DataFrame para mantener solo esas columnas
tpv_filtrado = tpv[columnas_a_mantener]

missing_values = tpv_filtrado.isna().sum()
missing_values = missing_values[missing_values != 0]
print(missing_values)

def detectar_anomalias(df):
    resultados = {}

    for columna in df.columns:
        # Ignorar columnas que son completamente numéricas o categóricas
        tipos_unicos = df[columna].apply(type).value_counts(normalize=True)

        if len(tipos_unicos) > 1:  # Si hay más de un tipo de dato
            tipo_dominante = tipos_unicos.idxmax()  # Tipo dominante
            porcentaje_dominante = tipos_unicos.max() * 100  # Porcentaje del tipo dominante

            # Si el porcentaje del tipo dominante es alto pero no 100%
            if porcentaje_dominante > 90:
                resultados[columna] = {
                    'tipo_dominante': tipo_dominante,
                    'porcentaje_dominante': porcentaje_dominante,
                    'tipos_unicos': tipos_unicos
                }

    return pd.DataFrame(resultados).T

# Aplicar la función al DataFrame
anomalies = detectar_anomalias(tpv_filtrado)
print(anomalies)

# Lista de columnas que deben ser numéricas
columnas_numericas = [
    'numero_total_de_garajes', 'total_cupos_parquedaro', 'numero_total_depositos', 'area_privada',
    'valor_area_privada', 'area_garaje', 'valor_area_garaje', 'area_deposito', 'valor_area_deposito',
    'area_terreno', 'valor_area_terreno', 'area_construccion', 'valor_area_construccion', 'area_otros',
    'valor_area_otros', 'valor_area_libre', 'valor_total_avaluo',
    'area_valorada', 'numero_piso', 'numero_de_edificios', 'vetustez', 'pisos_bodega', 'habitaciones',
    'estar_habitacion', 'cuarto_servicio', 'closet', 'sala', 'comedor', 'bano_privado', 'bano_social',
    'bano_servicio', 'cocina', 'estudio', 'balcon', 'terraza', 'patio_interior', 'jardin',
    'zona_de_ropas', 'zona_verde_privada', 'local', 'oficina', 'bodega', 'contadores_agua',	'contadores_luz',
    'unidades','observaciones_aislamiento_posterior','observaciones_aislamiento_lateral','observaciones_antejardin',
    'observaciones_indice_construccion','predio_subdividido_fisicamente'
]

# Contador para las columnas cambiadas
columnas_cambiadas = 0

# Convertir las columnas a tipo numérico y luego imputar valores faltantes o no numéricos
for col in columnas_numericas:
    original_non_numeric = tpv_filtrado[col].apply(lambda x: not pd.api.types.is_numeric_dtype(type(x))).sum()
    tpv_filtrado.loc[:, col] = pd.to_numeric(tpv_filtrado[col], errors='coerce')  # Convertir a numérico, convirtiendo no numéricos en NaN
    median = tpv_filtrado[col].median()  # Calcular la mediana de la columna
    tpv_filtrado.loc[:, col].fillna(median, inplace=True)  # Reemplazar NaN (que fueron originalmente no numéricos) con la mediana

    # Verificar si hubo un cambio en la columna
    converted_non_numeric = tpv_filtrado[col].isna().sum()
    if original_non_numeric > 0 or converted_non_numeric > 0:
        columnas_cambiadas += 1

# Verificar la conversión y notificar el resultado
correct_conversion = all(tpv_filtrado[columnas_numericas].dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)))

if correct_conversion:
    print(f"Todas las columnas se convirtieron e imputaron correctamente. Se cambiaron {columnas_cambiadas} columnas.")
else:
    print(f"Hubo un error en la conversión o imputación. Se cambiaron {columnas_cambiadas} columnas.")

lista_numericas = tpv_filtrado.select_dtypes(include=['number']).columns.tolist()
print("Variables numéricas:", lista_numericas)
tpv_filtrado.head()

"""**Imputar Variables Numericas Interpolate**"""

tpv_filtrado.interpolate(inplace=True)
tpv_filtrado.isna().sum()
tpv_filtrado.duplicated().sum()
tpv_filtrado.loc[tpv_filtrado.duplicated().sum()]

def validar_tipo_columna(col):
    """Valida el tipo de dato predominante en la columna."""
    tipo_dominante = col.apply(type).mode()[0]  # Encuentra el tipo de dato más común
    col_coherente = col.apply(lambda x: x if isinstance(x, tipo_dominante) else None)  # Reemplaza valores incorrectos por None
    return col_coherente, tipo_dominante

def clasificar_columna(col):
    """Clasifica la columna como numérica, categórica o factor."""
    tipo_dominante = col.apply(type).mode()[0]
    if pd.api.types.is_numeric_dtype(tipo_dominante):
        return 'numérica'
    else:
        return 'categórica'

def limpiar_y_clasificar(df):
    """Limpia las columnas, elimina valores incorrectos y clasifica cada columna."""
    tipo_columnas = {}

    for col in df.columns:
        # Validar y limpiar la columna
        df[col], tipo_dominante = validar_tipo_columna(df[col])

        # Convertir a numérico si es posible
        if pd.api.types.is_numeric_dtype(tipo_dominante):
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Clasificar la columna
        tipo_columnas[col] = clasificar_columna(df[col])

    return tipo_columnas

# Aplicar la función a tu DataFrame filtrado
tipo_columnas = limpiar_y_clasificar(tpv_filtrado)

# Mostrar la clasificación
for col, tipo in tipo_columnas.items():
    print(f"Columna '{col}' es de tipo: {tipo}")

# Listar las columnas por tipo
columnas_numericas = [col for col, tipo in tipo_columnas.items() if tipo == 'numérica']
columnas_categoricas = [col for col, tipo in tipo_columnas.items() if tipo == 'categórica']
#Combertir en Mayuscula
for col in columnas_categoricas:
    tpv_filtrado[col] = tpv_filtrado[col].str.upper()

print("Columnas numéricas:", columnas_numericas)
print("Columnas categóricas:", columnas_categoricas)

# Listas para guardar las columnas que cumplen o no cumplen con el 80% de 'SI', 'NO' o '0'
columnas_si_no = []
columnas_no_cumplen = []

# Recorrer todas las columnas categóricas para verificar sus valores
for col in columnas_categoricas:
    # Convertir la columna a mayúsculas para asegurar consistencia
    tpv_filtrado[col] = tpv_filtrado[col].str.upper()

    # Contar el total de valores en la columna
    total_values = len(tpv_filtrado[col])

    # Contar cuántos valores son 'SI', 'NO', o '0'
    valid_values = tpv_filtrado[col].isin(['SI', 'NO']).sum()

    # Validar si al menos el 80% de los valores son 'SI', 'NO', o '0'
    if valid_values / total_values >= 0.5:
        columnas_si_no.append(col)
    else:
        columnas_no_cumplen.append(col)

# Filtrar y limpiar las columnas que cumplen
for col in columnas_si_no:
    # Convertir '0' a 'NO'
    tpv_filtrado[col] = tpv_filtrado[col].replace('0', 'NO')

    # Eliminar filas con valores incorrectos
    tpv_filtrado = tpv_filtrado[tpv_filtrado[col].isin(['SI', 'NO'])]

# Verificar el resultado
print(f"Columnas que cumplen con el 80% de 'SI', 'NO' o '0': {columnas_si_no}")
print(f"Columnas que no cumplen con el 80% de 'SI', 'NO' o '0': {columnas_no_cumplen}")

# Mostrar algunas filas del DataFrame limpio para verificar
print(tpv_filtrado[columnas_si_no].head())

pd.set_option('display.max_rows', None)
# Lista para guardar la información de categorías y sus frecuencias
lista_categorias = []

# Recorrer todas las columnas categóricas
for col in columnas_no_cumplen:
    # Contar la cantidad de datos por categoría
    conteo_categorias = tpv_filtrado[col].value_counts()

    # Si la columna tiene menos de 10 categorías, agregar a la lista
    if len(conteo_categorias) < 10:
        for categoria, conteo in conteo_categorias.items():
            lista_categorias.append({
                'Columna': col,
                'Categoría': categoria,
                'Frecuencia': conteo
            })

# Convertir la lista en un DataFrame para visualizarla como una tabla
tabla_categorias = pd.DataFrame(lista_categorias)

# Mostrar la tabla
print(tabla_categorias)

Prueba = tpv_filtrado

tpv_filtrado = Prueba
tpv_filtrado.shape

# Ahora Transformamos lo datos que estan en 0 y no s epuedes relacionar con otros

# Omitir filas con la categoría '0' en las columnas especificadas
columnas_omit_0 = ['motivo', 'proposito', 'tipo_credito', 'tipo_inmueble', 'ocupante', 'aislamiento_posterior']
tpv_filtrado = tpv_filtrado[~tpv_filtrado[columnas_omit_0].isin(['0']).any(axis=1)]

# Omitir filas con las categorías '1' y '0' en la columna 'estructura'
tpv_filtrado = tpv_filtrado[~tpv_filtrado['estructura'].isin(['0', '1'])]

# Omitir filas con la categoría '2' en la columna 'ajustes_sismoresistentes'
tpv_filtrado = tpv_filtrado[~tpv_filtrado['ajustes_sismoresistentes'].isin(['2'])]

# Omitir filas con las categorías 'COMERCIAL', 'INDUSTRIAL', 'OFICINA' en la columna 'estrato'
tpv_filtrado = tpv_filtrado[~tpv_filtrado['estrato'].isin(['COMERCIAL', 'INDUSTRIAL', 'OFICINA'])]

# Omitir filas que no contengan las categorías 'SENCILLO', 'SIN ACABADOS', 'LUJOSO', 'BUENO' en la columna 'calidad_acabados_pisos'
tpv_filtrado = tpv_filtrado[tpv_filtrado['calidad_acabados_pisos'].isin(['SENCILLO', 'SIN ACABADOS', 'LUJOSO', 'BUENO','NORMAL'])]

# Transformar la categoría '0' a 'NO APLICA' en las columnas especificadas
columnas_transform_no_aplica = ['aislamiento_lateral', 'antejardin', 'indice_ocupacion', 'indice_construccion', 'tipo_vigilancia', 'tipo_deposito','altura_permitida']
tpv_filtrado[columnas_transform_no_aplica] = tpv_filtrado[columnas_transform_no_aplica].replace('0', 'NO APLICA')

# Transformar la categoría '0' a 'OTRO' en las columnas especificadas
columnas_transform_otro = ['condicion_ph', 'clase_inmueble', 'uso_actual']
tpv_filtrado[columnas_transform_otro] = tpv_filtrado[columnas_transform_otro].replace('0', 'OTRO')

# Lista para guardar columnas con más de 10 categorías
columnas_mas_de_10_categorias = []

# Recorrer todas las columnas categóricas
for col in columnas_no_cumplen:
    # Contar la cantidad de categorías únicas en la columna
    num_categorias = tpv_filtrado[col].nunique()

    # Si la columna tiene más de 10 categorías, agregar a la lista
    if num_categorias > 20:
        columnas_mas_de_10_categorias.append({
            'Columna': col,
            'Cantidad de Categorías': num_categorias
        })

# Convertir la lista en un DataFrame para visualizarla como una tabla
tabla_categorias_mas_de_10 = pd.DataFrame(columnas_mas_de_10_categorias)

# Mostrar la tabla
print(tabla_categorias_mas_de_10)

# Definir el umbral de longitud para considerar un nombre de categoría como "muy largo"
longitud_maxima = 20  # Ajusta este valor según lo que consideres "muy largo"

# Columnas que no se deben procesar
columnas_excluidas = [
    'departamento_inmueble',
    'municipio_inmueble',
    'barrio',
    'direccion_inmueble_informe','Longitud', 'Latitud'
]

# Crear un DataFrame vacío para almacenar los datos depurados
datos_depurados = pd.DataFrame()
# Recorrer todas las columnas categóricas excepto las excluidas
for col in [c for c in columnas_categoricas if c not in columnas_excluidas]:
    # Filtrar las categorías con nombres muy largos
    categorias_largas = tpv_filtrado[col].apply(lambda x: len(str(x)) > longitud_maxima)

    # Contar la cantidad de datos por categoría
    conteo_categorias = tpv_filtrado[col].value_counts()

    # Encontrar las categorías con nombres largos y menos de 10 datos
    categorias_a_omitir = conteo_categorias[(conteo_categorias < 10) & categorias_largas]

    # Filtrar los datos a depurar
    datos_a_depurar = tpv_filtrado[tpv_filtrado[col].isin(categorias_a_omitir.index)]

    # Agregar los datos depurados al DataFrame
    datos_depurados = pd.concat([datos_depurados, datos_a_depurar])

    # Omitir los datos que cumplen estas condiciones en el DataFrame original
    tpv_filtrado = tpv_filtrado[~tpv_filtrado[col].isin(categorias_a_omitir.index)]

# Crear un nuevo DataFrame excluyendo las columnas especificadas
nuevo_dataframe = tpv_filtrado.drop(columns=columnas_excluidas)

# Verificar el resultado
print("Datos filtrados y nuevo DataFrame creado correctamente.")
print(f"Total de filas en 'datos_depurados': {len(datos_depurados)}")
print(f"Columnas en el nuevo DataFrame: {nuevo_dataframe.columns.tolist()}")

# Mostrar algunas filas del nuevo DataFrame
print(nuevo_dataframe.head())

print("Columnas numéricas:", columnas_numericas)
print("Columnas categóricas:", columnas_categoricas)
nuevo_dataframe.to_csv('df_limpio.csv', index=True)

"""**ANÁLISIS UNIVARIADO**



*   Tablas de frecuencias
*   Medidas descriptivas
*   Gráficas

**Tablas de frecuencias - variables categoricas**
"""

# Recorrer todas las columnas categóricas para generar tablas de frecuencias
tablas_frecuencias = {}

for col in columnas_categoricas:
    if col in nuevo_dataframe.columns:
        # Generar la tabla de frecuencias para la columna
        tabla_frecuencia = nuevo_dataframe[col].value_counts().reset_index()

        # Renombrar las columnas para claridad
        tabla_frecuencia.columns = [col, 'Frecuencia']

        # Guardar la tabla en un diccionario
        tablas_frecuencias[col] = tabla_frecuencia
    else:
        print(f"Advertencia: La columna '{col}' no existe en 'nuevo_dataframe' y será omitida.")

# Mostrar algunas tablas de frecuencias
for col, tabla in tablas_frecuencias.items():
    print(f"Tabla de frecuencias para la columna '{col}':")
    print(tabla)
    print("\n" + "="*50 + "\n")

nuevo_dataframe.shape

# Reemplazar NA en columnas numéricas con la mediana
for col in columnas_numericas:
    median = tpv_filtrado[col].median()
    tpv_filtrado[col].fillna(median, inplace=True)

# Reemplazar valores 0 en columnas numéricas con la mediana, si es necesario
for col in columnas_numericas:
    if (tpv_filtrado[col] == 0).sum() > 0:  # Si hay valores 0
        tpv_filtrado[col].replace(0, median, inplace=True)

# Verificar si quedan NA
na_count = tpv_filtrado[columnas_numericas].isna().sum()
print("Valores NA restantes por columna:")
print(na_count[na_count > 0])

# Verificar si quedan valores 0
zero_count = (tpv_filtrado[columnas_numericas] == 0).sum()
print("Valores 0 restantes por columna:")
print(zero_count[zero_count > 0])

"""**Preparar los datos para el ACM**"""

columnas_categoricas = [
    'motivo', 'proposito', 'tipo_avaluo', 'tipo_credito', 'sector',
    'alcantarillado_en_el_sector', 'acueducto_en_el_sector', 'gas_en_el_sector',
    'energia_en_el_sector', 'telefono_en_el_sector', 'vias_pavimentadas',
    'sardineles_en_las_vias', 'andenes_en_las_vias', 'estrato', 'barrio_legal',
    'topografia_sector', 'condiciones_salubridad', 'transporte', 'demanda_interes',
    'paradero', 'alumbrado', 'arborizacion', 'alamedas', 'ciclo_rutas',
    'nivel_equipamiento_comercial', 'alcantarillado_en_el_predio',
    'acueducto_en_el_predio', 'gas_en_el_predio', 'energia_en_el_predio',
    'telefono_en_el_predio', 'tipo_inmueble', 'uso_actual', 'clase_inmueble',
    'ocupante', 'sometido_a_propiedad_horizontal', 'altura_permitida',
    'aislamiento_posterior', 'aislamiento_lateral', 'antejardin',
    'indice_ocupacion', 'indice_construccion', 'accesorios', 'condicion_ph', 'rph',
    'porteria', 'citofono', 'bicicletero', 'piscina', 'tanque_de_agua',
    'club_house', 'garaje_visitantes', 'teatrino', 'sauna', 'vigilancia_privada',
    'tipo_vigilancia', 'administracion', 'estructura', 'ajustes_sismoresistentes',
    'cubierta', 'fachada', 'tipo_fachada', 'estructura_reforzada', 'danos_previos',
    'material_de_construccion', 'iluminacion', 'ventilacion', 'irregularidad_planta',
    'irregularidad_altura', 'calidad_acabados_pisos', 'estado_acabados_muros',
    'calidad_acabados_muros', 'estado_acabados_techos', 'calidad_acabados_techos',
    'estado_acabados_madera', 'calidad_acabados_madera', 'estado_acabados_metal',
    'calidad_acabados_metal', 'estado_acabados_banos', 'calidad_acabados_banos',
    'estado_acabados_cocina', 'calidad_acabados_cocina', 'tipo_garaje',
    'tipo_deposito', 'area_libre'
]

df_categoricas = nuevo_dataframe[columnas_categoricas]

"""**Realizar el ACM**"""

# Crear el objeto MCA (Multiple Correspondence Analysis) con el motor correcto
acm = prince.MCA(
    n_components=2,  # Número de dimensiones
    n_iter=3,
    copy=True,
    check_input=True,
    engine='sklearn',  # Usamos sklearn como motor para evitar el error
    random_state=42
)

# Ajustar el modelo a los datos y transformar
acm = acm.fit(df_categoricas)

# Transformar los datos
df_acm = acm.transform(df_categoricas)

"""**Visualización de los resultados**"""

# Obtener las coordenadas de las filas
fila_coordenadas = acm.row_coordinates(df_categoricas)

# Obtener las coordenadas de las columnas
columna_coordenadas = acm.column_coordinates(df_categoricas)


# Crear una figura y ejes
plt.figure(figsize=(10, 7))

# Graficar las coordenadas de las filas
plt.scatter(fila_coordenadas[0], fila_coordenadas[1], alpha=0.5, label='Filas')

# Graficar las coordenadas de las columnas
plt.scatter(columna_coordenadas[0], columna_coordenadas[1], alpha=0.75, c='red', label='Columnas')

# Añadir etiquetas a las columnas
for i, col in enumerate(columna_coordenadas.index):
    plt.text(columna_coordenadas[0][i], columna_coordenadas[1][i], str(col), color='red', fontsize=12)

# Añadir título y leyenda
plt.title('Análisis de Correspondencias Múltiples (ACM)')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()

# Realizar el ACM
acm = prince.MCA(n_components=2, engine='sklearn', random_state=42).fit(df_categoricas)

# Obtener la contribución de las variables a los componentes
contribucion_columnas = acm.column_contributions_

# Crear un DataFrame para visualizar mejor la contribución
contribucion_df = pd.DataFrame(contribucion_columnas, index=df_categoricas.columns)
contribucion_df.columns = ['Contribución Componente 1', 'Contribución Componente 2']

# Ordenar el DataFrame por la contribución al primer componente
contribucion_df_sorted = contribucion_df.sort_values(by='Contribución Componente 1', ascending=False)

# Mostrar las variables con mayor contribución al primer componente
print(contribucion_df_sorted.head(10))  # Mostrar las 10 principales