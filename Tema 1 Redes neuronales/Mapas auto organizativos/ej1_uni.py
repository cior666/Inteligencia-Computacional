import numpy as np
import matplotlib.pyplot as plt
from ej1 import neurona_ganadora

def entrenar_som_1d(datos, cantidad_neuronas):
    pesos = np.random.uniform(-0.5,0.5,size=(cantidad_neuronas, 2))
    posiciones = np.arange(cantidad_neuronas)
    # ETAPA 1
    epocas_1 = 100
    radio_1 = cantidad_neuronas / 2
    tasa1 = 0.5
    for epoca in range(epocas_1):
        indices = np.random.permutation(len(datos))
        for indice in indices:
            x = datos[indice]
            ganadora = neurona_ganadora(x, pesos)
            for j in range(cantidad_neuronas):
                distancia = abs(posiciones[j] - posiciones[ganadora])
                if distancia <= radio_1:    
                    pesos[j] += tasa1 * (x - pesos[j])

    graficar_som_1d(datos,pesos,1)
    # ETAPA 2
    epocas_2 = 100
    radio_inicial = radio_1
    radio_final = 1
    tasa_inicial = tasa1
    tasa_final = 0.1

    for epoca in range(epocas_2):
        radio = radio_inicial + (radio_final - radio_inicial) * epoca / (epocas_2 - 1)
        tasa = tasa_inicial + (tasa_final - tasa_inicial) * epoca / (epocas_2 - 1)
        indices = np.random.permutation(len(datos))
        for indice in indices:
            x = datos[indice]
            ganadora = neurona_ganadora(x, pesos)
            for j in range(cantidad_neuronas):
                distancia = abs(posiciones[j] - posiciones[ganadora])
                if distancia <= radio:
                    pesos[j] += tasa * (x - pesos[j])

    graficar_som_1d(datos,pesos,2)

    # ETAPA 3
    epocas_3 = 300
    radio = 0
    tasa = 0.05
    for epoca in range(epocas_3):
        indices = np.random.permutation(len(datos))
        for indice in indices:
            x = datos[indice]
            ganadora = neurona_ganadora(x, pesos)
            pesos[ganadora] += tasa * (x - pesos[ganadora])
    graficar_som_1d(datos,pesos,3)
    return pesos

def graficar_som_1d(datos, pesos, etapa):
    plt.figure(figsize=(8, 8))
    ganadoras = np.array([neurona_ganadora(x, pesos)for x in datos])
    plt.scatter(datos[:, 0],datos[:, 1],c=ganadoras,cmap="tab20",s=10)
    plt.scatter(pesos[:, 0],pesos[:, 1],marker="x",color="black",s=50)
    for i in range(len(pesos) - 1):
        p1 = pesos[i]
        p2 = pesos[i + 1]
        plt.plot([p1[0], p2[0]],[p1[1], p2[1]],"k-",linewidth=0.8)
    plt.title("SOM 1D - Etapa " + str(etapa))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid()



T = np.loadtxt("te.csv", delimiter=",")
cantidad_neuronas = 100
print("Entrenando SOM 1D con T...")
pesos_T_1d = entrenar_som_1d(T,cantidad_neuronas)
plt.show()