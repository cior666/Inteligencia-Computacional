import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# cargar los datos

trn = np.loadtxt(r"C:\Users\conra\OneDrive\Desktop\Facu Conrado\CUARTO AÑO\Inteligencia Computacional\Tema 1 Redes neuronales\Mapas auto organizativos\iris81_trn.csv",delimiter=",")
# separar datos y clases
# las primeras 4 columnas son las caracteristicas de iris
datos = trn[:, :4]
# las ultimas 3 columnas indican la clase
clases_codigo = trn[:, 4:]
# convertimos el codigo de tres valores a un numero de clase
# 0 = setosa
# 1 = versicolor
# 2 = virginica

#queda igual q el ej anterior
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
        distancias = np.linalg.norm(datos[:, np.newaxis, :] -centroides[np.newaxis, :, :],axis=2)
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
# guardamos el silhouette de cada valor de k
valores_silhouette = []
#lo q hace el silhouete es medir que tan bien separado y compacto esta cada agrupamiento, buscamos q sea lo mas grande posible
#aca hay 3 casos, si esta cerca de 1 el patron esta bien dentro de su cluster y separado de los demas
#si es cercano a 0 el patron esta en una zona de separacion de clusters
#si es menor q 0 deberia de ser asignado a otro cluster 

for k in valores_k:
    # entrenamos k-medias con el valor de k actual
    centroides, grupos = k_medias(datos, k)
    # calculamos la metrica silhouette
    valor = silhouette_score(
        datos,
        grupos
    )
    # guardamos el resultado
    valores_silhouette.append(valor)
    print(
        "k =", k,
        "-> silhouette =", valor
    )

# buscar el k optimo
# buscamos la posicion donde esta el mayor silhouette
indice_mejor = np.argmax(valores_silhouette)
# obtenemos el valor de k correspondiente
k_optimo = list(valores_k)[indice_mejor]
# obtenemos el mejor valor de silhouette
mejor_silhouette = valores_silhouette[indice_mejor]
print("\nresultado:")
print("k optimo:", k_optimo)
print("mejor silhouette:", mejor_silhouette)
# graficar los resultados
plt.figure(figsize=(7, 5))

plt.plot(
    valores_k,
    valores_silhouette,
    marker="o"
)

plt.xlabel("cantidad de clusters k")
plt.ylabel("silhouette score")
plt.title("silhouette score para distintos valores de k")

plt.grid()
plt.show()