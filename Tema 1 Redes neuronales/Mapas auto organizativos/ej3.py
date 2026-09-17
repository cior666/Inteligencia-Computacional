import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import davies_bouldin_score


# cargar los datos

trn = np.loadtxt(
    r"C:\Users\conra\OneDrive\Desktop\Facu Conrado\CUARTO AÑO\Inteligencia Computacional\Tema 1 Redes neuronales\Mapas auto organizativos\iris81_trn.csv",
    delimiter=","
)
# separar datos y clases
# las primeras 4 columnas son las caracteristicas de iris
datos = trn[:, :4]
# las ultimas 3 columnas indican la clase
clases_codigo = trn[:, 4:]
# convertimos el codigo de tres valores a un numero de clase
# 0 = setosa
# 1 = versicolor
# 2 = virginica
# queda igual q el ej anterior
clases = np.full(len(clases_codigo), -1)
clases[np.all(clases_codigo == [-1, -1, 1], axis=1)] = 0
clases[np.all(clases_codigo == [-1, 1, -1], axis=1)] = 1
clases[np.all(clases_codigo == [1, -1, -1], axis=1)] = 2
nombres_clases = ["setosa", "versicolor", "virginica"]

# funcion k-medias
def k_medias(datos, k, max_epocas=100):
    # elegimos k patrones al azar para usar como centroides iniciales
    indices = np.random.choice(len(datos),k,replace=False)
    centroides = datos[indices].copy()
    for epoca in range(max_epocas):
        # calculamos la distancia de cada patron a cada centroide
        distancias = np.linalg.norm(
            datos[:, np.newaxis, :] -
            centroides[np.newaxis, :, :],
            axis=2
        )
        # cada patron se asigna al centroide mas cercano
        grupos = np.argmin(distancias, axis=1)
        # guardamos los centroides anteriores
        centroides_anterior = centroides.copy()
        # actualizamos cada centroide
        for j in range(k):
            patrones_grupo = datos[grupos == j]
            # si el grupo tiene patrones, calculamos su promedio
            if len(patrones_grupo) > 0:
                centroides[j] = np.mean(
                    patrones_grupo,
                    axis=0
                )
        # si los centroides dejaron de cambiar, terminamos
        if np.allclose(
            centroides,
            centroides_anterior
        ):
            break
    return centroides, grupos

# probar distintos valores de k
# probamos valores de k desde 2 hasta 10
valores_k = range(2, 11)
# hacemos varias ejecuciones para evitar q un resultado dependa demasiado de la inicializacion aleatoria
repeticiones = 10
# guardamos el rendimiento promedio de cada metrica
rendimiento_ari = []
rendimiento_silhouette = []
rendimiento_davies = []

for k in valores_k:
    # guardamos los resultados de las 10 repeticiones
    scores_ari = []
    scores_silhouette = []
    scores_davies = []
    for i in range(repeticiones):
        # entrenamos k-medias con el valor de k actual
        centroides, grupos = k_medias(datos, k)
        # ARI compara los grupos encontrados con las clases reales es una metrica externa
        valor_ari = adjusted_rand_score(clases,grupos)
        # silhouette mide q tan compactos y separados estan los grupos buscamos q sea lo mas grande posible
        valor_silhouette = silhouette_score(datos,grupos)
        # davies-bouldin tambien mide la separacion de los grupos en este caso buscamos un valor lo mas chico posible
        valor_davies = davies_bouldin_score(datos,grupos)
        # guardamos los resultados de esta repeticion
        scores_ari.append(valor_ari)
        scores_silhouette.append(valor_silhouette)
        scores_davies.append(valor_davies)
    # calculamos el promedio de las 10 repeticiones
    rendimiento_ari.append(np.mean(scores_ari))
    rendimiento_silhouette.append(np.mean(scores_silhouette))
    rendimiento_davies.append(np.mean(scores_davies))

    print("k =", k,"-> ari =", rendimiento_ari[-1],"silhouette =", rendimiento_silhouette[-1],"davies-bouldin =", rendimiento_davies[-1])
# buscar el k optimo
# para ari buscamos el mayor valor
indice_ari = np.argmax(rendimiento_ari)
# para silhouette buscamos el mayor valor
indice_silhouette = np.argmax(rendimiento_silhouette)
# para davies-bouldin buscamos el menor valor
indice_davies = np.argmin(rendimiento_davies)
# obtenemos el k correspondiente a cada metrica
k_optimo_ari = list(valores_k)[indice_ari]

k_optimo_silhouette = list(valores_k)[indice_silhouette]

k_optimo_davies = list(valores_k)[indice_davies]
# obtenemos los mejores valores
mejor_ari = rendimiento_ari[indice_ari]
mejor_silhouette = rendimiento_silhouette[indice_silhouette]

mejor_davies = rendimiento_davies[indice_davies]
# mostramos los resultados
print("\nresultados:")
print("ari -> k optimo:",k_optimo_ari,"score:",mejor_ari)
print("silhouette -> k optimo:",k_optimo_silhouette,"score:",mejor_silhouette)
print("davies-bouldin -> k optimo:",k_optimo_davies,"score:",mejor_davies)


# graficar ari
plt.figure(figsize=(7, 5))
plt.plot(
    valores_k,
    rendimiento_ari,
    marker="o"
)
plt.xlabel("cantidad de clusters k")
plt.ylabel("adjusted rand index")
plt.title("ari para distintos valores de k")
plt.grid()
plt.show()


# graficar silhouette
plt.figure(figsize=(7, 5))
plt.plot(
    valores_k,
    rendimiento_silhouette,
    marker="o"
)
plt.xlabel("cantidad de clusters k")
plt.ylabel("silhouette score")
plt.title("silhouette para distintos valores de k")
plt.grid()
plt.show()

# graficar davies-bouldin
plt.figure(figsize=(7, 5))
plt.plot(
    valores_k,
    rendimiento_davies,
    marker="o"
)
plt.xlabel("cantidad de clusters k")
plt.ylabel("davies-bouldin index")
plt.title("davies-bouldin para distintos valores de k")
plt.grid()
plt.show()