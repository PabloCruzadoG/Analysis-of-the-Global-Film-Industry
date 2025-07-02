# ------------ WEB SCRAPING A CSV ------------
import requests
from bs4 import BeautifulSoup
import csv
import re

# URL de la página de Wikipedia con la lista de películas
url = 'https://es.wikipedia.org/wiki/Anexo:Pel%C3%ADculas_con_las_mayores_recaudaciones'

# Realizamos la petición HTTP a la página y obtenemos la respuesta
response = requests.get(url)

# Parseamos el contenido HTML de la página con BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Buscamos la tabla por su clase CSS, que contiene los datos que queremos extraer
table = soup.find('table', {'class': 'wikitable'})

# Lista para almacenar los datos de cada película
data = []

# Iteramos sobre cada fila de la tabla (excepto el encabezado)
for row in table.find_all('tr')[1:]:  # [1:] omite el encabezado de la tabla
    cols = row.find_all('td')  # Encuentra todas las columnas de la fila
    if len(cols) > 0:  # Si la fila tiene columnas (datos)

        # Extrae y limpia el texto de cada columna de interés
        pelicula = cols[0].get_text(strip=True)
        
        # Procesa la recaudación mundial para eliminar puntos como separadores de miles
        recaudacion_mundial_raw = cols[1].get_text(strip=True)
        # Elimina puntos como separadores de miles y cambia comas por puntos para decimales
        recaudacion_mundial = re.sub(r'\.(?=\d{3}(?:\D|$))', '', recaudacion_mundial_raw).replace(',', '.')
        # Extrae solo los números de la cadena procesada
        recaudacion_mundial_match = re.search(r'\d+', recaudacion_mundial)
        # Obtiene la recaudación como número o cero si no se encontró
        recaudacion_mundial = recaudacion_mundial_match.group(0) if recaudacion_mundial_match else '0'

        # Sigue un proceso similar para el presupuesto
        presupuesto_raw = cols[4].get_text(strip=True)
        presupuesto = re.sub(r'\.(?=\d{3}(?:\D|$))', '', presupuesto_raw).replace(',', '.')
        presupuesto_match = re.search(r'\d+', presupuesto)
        presupuesto = presupuesto_match.group(0) if presupuesto_match else '0'

        # Extrae el texto de la distribuidora y el año de estreno
        distribuidora = cols[5].get_text(strip=True)
        ano_estreno = cols[6].get_text(strip=True)

        # Añade los datos de la película a la lista de datos
        data.append([pelicula, recaudacion_mundial, presupuesto, distribuidora, ano_estreno])

# Define el nombre del archivo CSV donde se guardarán los datos
csv_file_path = 'movies_scrap.csv'

# Crea y abre el archivo CSV para escritura
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Escribe la cabecera del CSV
    writer.writerow(['Película', 'Recaudación Mundial', 'Presupuesto', 'Distribuidora', 'Año de Estreno'])
    # Escribe las filas de datos de las películas en el archivo CSV
    writer.writerows(data)

# Imprime un mensaje indicando que los datos se han guardado correctamente
print(f"Datos guardados correctamente en {csv_file_path}")



# ------------ ELIMINAR LOS 0 del CSV del webscrapìng ------------
import csv

# Ruta del archivo CSV
csv_file_path = 'movies_scrap.csv'

# Leer los datos del archivo CSV
with open(csv_file_path, newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    data = list(reader)

# Modificar los datos (eliminar ceros del principio)
for row in data[1:]:  # Ignorar la cabecera
    row[1] = row[1].lstrip('0')  # Recaudación Mundial
    row[2] = row[2].lstrip('0')  # Presupuesto

# Escribir de nuevo en el archivo CSV
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"Archivo CSV actualizado correctamente: {csv_file_path}")


# ------------ LIMPIEZA Y TRADUCCION ------------
import pandas as pd

# Cargando los archivos
movies_csv_path = 'movies.csv'
movies_scrap_csv_path = 'movies_scrap.csv'
movies_json_path = 'moviesr.json'

# Leyendo los archivos CSV y JSON
movies_df = pd.read_csv(movies_csv_path)
movies_scrap_df = pd.read_csv(movies_scrap_csv_path)
movies_json_df = pd.read_json(movies_json_path)

# Traduciendo las columnas de 'movies.csv' y 'moviesr.json' al español
movies_df.columns = ['Titulo', 'Clasificacion', 'Genero', 'Ano', 'Lanzamiento', 'Puntuación', 'Votos', 
                     'Director', 'Escritor', 'Estrella', 'Pais', 'Presupuesto', 'Ingresos', 'Compania', 'Duracion']
movies_json_df.columns = ['Informacion de la Pelicula', 'Titulo', 'Descripcion de la Pelicula', 'Ano', 'Presupuesto',
                          'Apertura Domestica', 'Ventas Domesticas', 'Ventas Internacionales',
                          'Ventas Mundiales', 'Fecha de Lanzamiento', 'Genero', 'Tiempo de Ejecucion', 'Licencia',
                          'Distribuidor']


# Verificando si hay datos duplicados o nulos en los tres dataframes
print("Resultados de verificación de datos duplicados y nulos:")

print("\nPara movies.csv (movies_df):")
# Para movies_df
duplicados_movies_df = movies_df.duplicated().sum()
print(f"Datos duplicados: {duplicados_movies_df}")
nulos_movies_df = movies_df.isnull().sum().sum()
print(f"Datos nulos: {nulos_movies_df}")

print("\nPara movies_scrap.csv (movies_scrap_df):")
# Para movies_scrap_df
duplicados_movies_scrap_df = movies_scrap_df.duplicated().sum()
print(f"Datos duplicados: {duplicados_movies_scrap_df}")
nulos_movies_scrap_df = movies_scrap_df.isnull().sum().sum()
print(f"Datos nulos: {nulos_movies_scrap_df}")

print("\nPara moviesr.json (movies_json_df):")
# Para movies_json_df
duplicados_movies_json_df = movies_json_df.duplicated().sum()
print(f"Datos duplicados: {duplicados_movies_json_df}")
nulos_movies_json_df = movies_json_df.isnull().sum().sum()
print(f"Datos nulos: {nulos_movies_json_df}")

# Limpieza de datos
movies_df_clean = movies_df.drop_duplicates().dropna()
movies_scrap_df_clean = movies_scrap_df.drop_duplicates().dropna()
movies_json_df_clean = movies_json_df.drop_duplicates().dropna()

# Guardar los conjuntos de datos limpios
clean_movies_csv_path = 'c_movies.csv'
clean_movies_scrap_csv_path = 'c_movies_scrap.csv'
clean_movies_json_path = 'c_moviesr.json'

movies_df_clean.to_csv(clean_movies_csv_path, index=False)
movies_scrap_df_clean.to_csv(clean_movies_scrap_csv_path, index=False)
movies_json_df_clean.to_json(clean_movies_json_path, orient='records', lines=True)

clean_movies_csv_path, clean_movies_scrap_csv_path, clean_movies_json_path


# ------------ PRIMER MERGE ------------ 
import pandas as pd
from pymongo import MongoClient

# Conectar con el servidor MongoDB 
client = MongoClient('mongodb+srv://-----------------')

# Seleccionar la base de datos
db = client['MoviesNoSQL']

# Seleccionar la colección
collection = db['Movies']

# Extraer los datos excluyendo el campo '_id'
moviesm_data = list(collection.find({}, {'_id': 0}))

# Crear un DataFrame de pandas con los datos extraídos
df_MoviesNoSQL = pd.DataFrame(moviesm_data)

#Cargar y Leer el CSV del WebScraping
c_movies_scrap_path = 'c_movies_scrap.csv'
c_movies_scrap_df = pd.read_csv(c_movies_scrap_path)

# Renombrando las columnas para que coincidan
df_MoviesNoSQL.rename(columns={'RecaudacionMundial': 'Recaudacion'}, inplace=True)
c_movies_scrap_df.rename(columns={'Pelicula': 'Titulo', 'Recaudación Mundial': 'Recaudacion'}, inplace=True)

# Conservando solo las columnas relevantes
df_MoviesNoSQL = df_MoviesNoSQL[['Titulo', 'Presupuesto', 'Recaudacion']]
c_movies_scrap_df = c_movies_scrap_df[['Titulo', 'Presupuesto', 'Recaudacion']]

# Uniendo los DataFrames
combined1_df = pd.merge(df_MoviesNoSQL, c_movies_scrap_df, on='Titulo', how='inner')

# Calcula el promedio de las columnas 'Presupuesto' y 'Recaudacion'
combined1_df['Presupuesto_promedio'] = (combined1_df['Presupuesto_x'] + combined1_df['Presupuesto_y']) / 2
combined1_df['Recaudacion_promedio'] = (combined1_df['Recaudacion_x'] + combined1_df['Recaudacion_y']) / 2

# Conserva solo las columnas relevantes en el DataFrame resultante
combined1_df = combined1_df[['Titulo', 'Presupuesto_promedio', 'Recaudacion_promedio']]

combined1_df.to_csv('merge1.csv', index=False)


# ------------ Final Merge ------------ 
import pandas as pd

# Cargando los archivos
csv_file_path = 'c_movies.csv'
json_file_path = 'c_moviesr.json'
merge_csv_path = 'merge1.csv'

# Leyendo el archivo CSV
df_csv = pd.read_csv(csv_file_path)

# Leyendo el archivo JSON
df_json = pd.read_json(json_file_path, lines=True)

# Renombrando las columnas del DataFrame JSON para alinearlas con los otros DataFrames
df_json = df_json.rename(columns={'Titulo': 'Title', 'Presupuesto': 'Budget', 'Ventas Mundiales': 'BoxOffice'})

# Seleccionando las columnas relevantes del DataFrame JSON
df_json_relevant = df_json[['Title', 'Budget', 'BoxOffice']]

# Leyendo el tercer archivo CSV
df_merge = pd.read_csv(merge_csv_path)

# Combinando df_csv con df_merge en 'Titulo'
merged_df_1 = df_csv.merge(df_merge, on='Titulo', how='inner')

# Combinando el resultado anterior con df_json_relevant en 'Titulo' (Title en df_json_relevant)
merged_df_final = merged_df_1.merge(df_json_relevant, left_on='Titulo', right_on='Title', how='inner')

# Convirtiendo todas las columnas de presupuesto y taquilla a numéricas, manejando entradas no numéricas
merged_df_final['Presupuesto'] = pd.to_numeric(merged_df_final['Presupuesto'], errors='coerce')
merged_df_final['Ingresos'] = pd.to_numeric(merged_df_final['Ingresos'], errors='coerce')
merged_df_final['Budget'] = pd.to_numeric(merged_df_final['Budget'], errors='coerce')
merged_df_final['BoxOffice'] = pd.to_numeric(merged_df_final['BoxOffice'], errors='coerce')

# Calculando el promedio de las columnas de presupuesto y taquilla
merged_df_final['avg_presupuesto'] = merged_df_final[['Presupuesto', 'Budget', 'Presupuesto_promedio']].mean(axis=1)
merged_df_final['avg_recaudacion'] = merged_df_final[['Ingresos', 'BoxOffice', 'Recaudacion_promedio']].mean(axis=1)

# Seleccionando solo las columnas relevantes para el resultado final
final_df = merged_df_final[['Titulo', 'avg_presupuesto', 'avg_recaudacion']]

final_df.to_csv('mergefinal.csv', index=False)


# ------------ VISUALES ------------
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo CSV
movies_df = pd.read_csv('c_movies.csv')

# 1. Cantidad de Películas por País - Choropleth Map

# Preparar datos para el Choropleth Map
country_counts = movies_df['Pais'].value_counts().reset_index()
country_counts.columns = ['Pais', 'Cantidad']

# Crear un Choropleth Map mostrando el número de películas producidas por cada país
choropleth_map = px.choropleth(country_counts, 
                               locations="Pais", 
                               locationmode='country names',
                               color="Cantidad", 
                               hover_name="Pais", 
                               color_continuous_scale=px.colors.sequential.Plasma,
                               title="Cantidad de Películas por País")

choropleth_map.show()



# 2. Cantidad de Películas por Género a lo Largo de los Años - Gráfico de Área

# Cargar el archivo CSV
movies_df = pd.read_csv('c_movies.csv')

# Convertir la columna 'Ano' a tipo datetime para facilitar la agrupación por año
movies_df['Ano'] = pd.to_datetime(movies_df['Ano'], format='%Y')

# Agrupar por año y género y contar el número de películas
area_data = movies_df.groupby([movies_df['Ano'].dt.year, 'Genero']).size().unstack()

# Crear el gráfico de área
plt.figure(figsize=(12, 6))
area_data.plot(kind='area', stacked=True, ax=plt.gca())
plt.title('Cantidad de Películas por Género a lo Largo de los Años')
plt.xlabel('Año')
plt.ylabel('Cantidad de Películas')
plt.legend(title='Género')
plt.show()



# 3. Cantidad de Películas Producidas por Año

# Agrupar los datos por año y contar la cantidad de películas
movies_per_year = movies_df.groupby('Ano').size()

# Crear un gráfico de área
plt.fill_between(movies_per_year.index, movies_per_year.values)
plt.title('Cantidad de Películas Producidas por Año')
plt.xlabel('Año')
plt.ylabel('Cantidad de Películas')
plt.show()



# 4. Ingresos Totales de Películas por País - Bubble Map

# Agrupar por país y sumar los ingresos
country_revenue = movies_df.groupby('Pais')['Ingresos'].sum().reset_index()

# Crear un Bubble Map
bubble_map = px.scatter_geo(country_revenue,
                            locations="Pais",
                            locationmode='country names',
                            size="Ingresos",
                            hover_name="Pais",
                            title="Ingresos Totales de Películas por País",
                            size_max=50)

bubble_map.show()


# 5. Distribución de Películas por Género - Grafico de Pie

# Agrupar los datos por generos
genre_counts = movies_df['Genero'].value_counts()

# Creacion del grafico de pie
plt.figure(figsize=(10, 8))
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribución de Películas por Género')
plt.show()



# 6. Cantidad de Películas por Género y Clasificación - Clustered Bar Plot

# Agrupar los datos por 'Genero' y 'Clasificacion' y contar la cantidad de peliculas
clustered_data = movies_df.groupby(['Genero', 'Clasificacion']).size().reset_index(name='Cantidad')

# Crear el Clustered Bar Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Genero', y='Cantidad', hue='Clasificacion', data=clustered_data)
plt.title('Cantidad de Películas por Género y Clasificación')
plt.xlabel('Género')
plt.ylabel('Cantidad')
plt.xticks(rotation=45)
plt.legend(title='Clasificación')
plt.show()


# Con el CSV de la unión de los 4 Dataframes 
# Comparación de Presupuesto y Recaudación por Película - Gráfico de Barras Horizontal y 
# Relación entre Presupuesto y Recaudación de las Películas - Scatter Plot

# Cargar el archivo CSV
file_path = 'mergefinal.csv'
mergefinal_df = pd.read_csv(file_path)

# 1. Gráfico de Barras Horizontal
plt.figure(figsize=(12, 8))
plt.barh(mergefinal_df['Titulo'], mergefinal_df['avg_presupuesto'], label='Presupuesto Promedio')
plt.barh(mergefinal_df['Titulo'], mergefinal_df['avg_recaudacion'], left=mergefinal_df['avg_presupuesto'], 
         label='Recaudación Promedio')
plt.xlabel('Cantidad en Billones ($)')
plt.ylabel('Título de la Película')
plt.title('Comparación de Presupuesto y Recaudación por Película')
plt.legend()
plt.show()

# 2. Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(mergefinal_df['avg_presupuesto'], mergefinal_df['avg_recaudacion'])
plt.xlabel('Presupuesto Promedio ($)')
plt.ylabel('Recaudación Promedio ($)')
plt.title('Relación entre Presupuesto y Recaudación de las Películas')
plt.grid(True)
plt.show()


# ------------ EXPORTACION DEL CSV MERGEFINAL A UNA BASE DE DATOS SQL ------------ 

import pandas as pd
from sqlalchemy import create_engine

# Leer el archivo CSV en un DataFrame de pandas
df_csv = pd.read_csv('mergefinal.csv')

# Conectar a MySQL usando SQLAlchemy
# Asegúrate de reemplazar 'root', 'root', 'localhost', '3306' y 'Titulo' con tus propios detalles de conexión.
engine = create_engine('mysql+pymysql://----------')

# Exportar el DataFrame a MySQL
# movies es el nombre de la tabla que se deseas utilizar en la base de datos MySQL.
df_csv.to_sql(name='movies', con=engine, if_exists='replace', index=False)

# Cerrar la conexión con la base de datos
engine.dispose()