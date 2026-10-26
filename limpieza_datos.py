#Paso 1: Estructura, nombres y tipos de datos

# 1.1 Ver la radiografía inicial (columnas, tipos y nulos)
df.info()

# 1.2 Estandarizar nombres a snake_case (minúsculas y guiones bajos)
df.columns = (
    df.columns
    .str.strip()                                    # Quita espacios al principio y final
    .str.replace(r'(?<=[a-z])(?=[A-Z])', '_', regex=True) # De CamelCase a snake_case
    .str.lower()                                    # A minúsculas
)

# 1.3 Convertir fechas (de string a fecha real de Pandas)
df["invoice_date"] = pd.to_datetime(df["invoice_date"])

# 1.4 Convertir tipos numéricos (asegurar float o int)
df["unit_price"] = df["unit_price"].astype(float)
df["customer_id"] = df["customer_id"].astype(str)  # Los IDs mejor como texto

# 1.5 Limpiar textos (quitar espacios en blanco invisibles)
df["description"] = df["description"].str.strip().str.upper()

#Paso 2: Duplicados

# 2.1 Contar cuántas filas están exactamente repetidas
df.duplicated().sum()

# 2.2 Eliminar duplicados exactos (quedándose con la primera aparición)
df = df.drop_duplicates(keep="first")

# 2.3 Eliminar duplicados basados solo en una clave única (ej: ID de transacción)
df = df.drop_duplicates(subset=["invoice_no", "stock_code"], keep="first")

#Paso 3: Gestión de valores nulos (Missing Values)

# 3.1 Ver cuántos nulos hay por columna y su porcentaje
df.isna().sum()
df.isna().mean() * 100  # Porcentaje de nulos

# 3.2 Borrar filas donde falte el identificador clave (usando subset)
df = df.dropna(subset=["customer_id"])

# 3.3 Imputar variables numéricas (con la mediana para que no afecten los extremos)
mediana_edad = df["edad"].median()
df["edad"] = df["edad"].fillna(mediana_edad)

# 3.4 Imputar variables categóricas (con una etiqueta fija o la moda)
df["description"] = df["description"].fillna("DESCONOCIDO")

#Paso 4: Reglas de Negocio y Valores Imposibles

# 4.1 Filtros booleanos de sentido común (cantidades y precios positivos)
df = df[df["quantity"] > 0]
df = df[df["unit_price"] > 0]

# Filtrar múltiples condiciones a la vez con .query() (muy legible)
df = df.query("quantity > 0 and unit_price > 0")

df["Quantity"].describe() #(te da los mínimos y máximos)

# Contar cuántas hay:
(df["Quantity"] < 0).sum()
# Resultado: 10.624 filas negativas (un 1.96% del total)

# 4.2 Fechas coherentes (la entrega debe ser posterior a la compra)
df = df[df["fecha_entrega"] >= df["fecha_pedido"]]


# 4.3 Corregir valores mal escritos o sinónimos
df["country"] = df["country"].replace({
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "RSA": "South Africa"
})

#Paso 5: Outliers y Valores Extremos

# 5.1 Ver estadísticas rápidas (mínimos, máximos, percentiles)
df[["quantity", "unit_price"]].describe()


# 5.2 Método de Percentiles (descartar el 1% más extremo por arriba y por abajo)
limite_inferior = df["unit_price"].quantile(0.01)  # Percentil 1%
limite_superior = df["unit_price"].quantile(0.99)  # Percentil 99%

df = df[(df["unit_price"] >= limite_inferior) & (df["unit_price"] <= limite_superior)]

# 5.3 Método "Clip" (en vez de borrar la fila, recorta el valor al límite máximo)
df["unit_price"] = df["unit_price"].clip(lower=limite_inferior, upper=limite_superior)

#Igual que tenemos en  src/cleaning.py, en producción se empaqueta así de elegante:

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    """Aplica el pipeline de limpieza completo."""
    return (
        df
        .drop_duplicates()
        .pipe(standardize_column_names)
        .dropna(subset=["customer_id"])
        .query("quantity > 0 and unit_price > 0")
    )

#Si tienes una variable guardada en tu código y quieres usarla dentro del .query(),
#  solo le pones una arroba @ delante del nombre:



