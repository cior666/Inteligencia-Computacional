import numpy as np
import pandas as pd
import time

# CARGAR LOS DATOS
ruta = r"C:\Users\conra\OneDrive\Desktop\Facu Conrado\CUARTO AÑO\Inteligencia Computacional\Inteligencia Colectiva\gr17.csv"

# cargar la matriz de distancias
# D[i,j] representa la distancia entre la ciudad i y la ciudad j
D = pd.read_csv(ruta, header=None).values

# cantidad de ciudades
n = len(D)

print("Dimensiones de la matriz:", D.shape)

# SISTEMA DE HORMIGAS
def sistema_hormigas(D,n_hormigas=17,alpha=1,beta=1,rho=0.5,Q=1,max_iter=500,tipo_deposito="global",seed=0):
    # cantidad de ciudades
    n = len(D)

    # generador de numeros aleatorios cada corrida puede utilizar una semilla diferente
    # para obtener diferentes recorridos
    rng = np.random.default_rng(seed)

    # INFORMACION HEURISTICA
    # eta[i,j] representa la informacion heuristica
    #
    # eta[i,j] = 1 / D[i,j]
    #
    # una distancia menor produce una mayor informacion heuristica y por lo tanto hace mas atractivo ese camino
    eta = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            # evitamos dividir por cero
            if D[i, j] != 0:
                eta[i, j] = 1 / D[i, j]


    # INICIALIZACION DE FEROMONAS
    # sigma[i,j] representa la cantidad de feromona asociada al camino entre i y j
    # inicialmente utilizamos el mismo valor para todos los caminos para no favorecer ninguna ruta
    sigma = np.ones((n, n))

    # no tiene sentido tener feromona desde una ciudad
    # hacia ella misma
    np.fill_diagonal(sigma, 0)

    # MEJOR SOLUCION ENCONTRADA
    # al principio no tenemos ningun camino
    mejor_camino = None

    # comenzamos con una longitud infinita
    mejor_longitud = np.inf

    # historial de la mejor solucion encontrada
    # en cada iteracion
    historial = []

    # ITERACIONES
    for t in range(max_iter):

        # caminos construidos por las hormigas
        caminos = []

        # longitud de cada camino
        longitudes = []
        # CADA HORMIGA CONSTRUYE UN CAMINO

        for k in range(n_hormigas):

            # todas las hormigas comienzan en la ciudad 0
            camino = [0]

            # ciudades que todavia no fueron visitadas
            no_visitadas = set(range(1, n))

            # CONSTRUCCION DEL CAMINO
            while no_visitadas:
                # ciudad actual
                i = camino[-1]
                # ciudades que pueden ser visitadas
                candidatos = np.array(list(no_visitadas))
                # PROBABILIDAD DE TRANSICION
                # la probabilidad depende de:
                # feromona^alpha * heuristica^beta
                # alpha controla la importancia de la feromona
                # beta controla la importancia de la heuristica

                # calculamos la influencia de la feromona
                # y de la informacion heuristica
                numerador = (sigma[i, candidatos] ** alpha* eta[i, candidatos] ** beta)

                # calculamos la suma de los valores
                suma = np.sum(numerador)

#                si por problemas numericos todos los valores resultan ser cero, utilizamos una probabilidad uniforme
                if suma == 0:

                    probabilidades = np.ones(len(candidatos)) / len(candidatos)

                else:
                    probabilidades = numerador / suma
                    numerador = (sigma[i, candidatos] ** alpha* eta[i, candidatos] ** beta)

                # seleccionamos la siguiente ciudad
                j = rng.choice(candidatos,p=probabilidades)
                # agregamos la ciudad al camino
                camino.append(j)

                # eliminamos la ciudad de las no visitadas
                no_visitadas.remove(j)

                # DEPOSITO LOCAL
                # en el deposito local modificamos la feromona inmediatamente despues de utilizar un camino
                # esto permite incorporar informacion durante la construccion de los caminos

                if tipo_deposito == "local":
                    # cantidad de feromona que se deposita
                    # sobre el camino utilizado
                    deposito = Q / D[i, j]
                    sigma[i, j] += deposito
                    sigma[j, i] += deposito
            # VOLVER A LA CIUDAD INICIAL
            # el recorrido debe ser cerrado
            camino.append(0)
            # CALCULAR LONGITUD DEL CAMINO
            longitud = 0
            for q in range(len(camino) - 1):
                i = camino[q]
                j = camino[q + 1]
                longitud += D[i, j]
            # guardar camino y longitud
            caminos.append(camino)
            longitudes.append(longitud)
            # ACTUALIZAR MEJOR SOLUCION GLOBAL
            if longitud < mejor_longitud:
                mejor_longitud = longitud
                mejor_camino = camino.copy()
        # EVAPORACION DE FEROMONAS
        # una parte de la feromona desaparece en cada iteracion
        # rho controla la tasa de evaporacion
        # rho grande -> mayor evaporacion
        # rho pequeño -> menor evaporacion
        sigma *= (1 - rho)
        sigma = np.maximum(sigma, 1e-12)

        # DEPOSITO DE FEROMONAS
        if tipo_deposito == "global":
            # DEPOSITO GLOBAL
            # todas las hormigas depositan feromona
            # despues de haber construido su recorrido los caminos mas cortos reciben mas feromona

            for camino, longitud in zip(caminos, longitudes):
                deposito = Q / longitud
                for q in range(len(camino) - 1):
                    i = camino[q]
                    j = camino[q + 1]
                    sigma[i, j] += deposito
                    sigma[j, i] += deposito

        elif tipo_deposito == "uniforme":
            # DEPOSITO UNIFORME
            # todas las rutas utilizadas reciben la misma cantidad de feromona, independientemente de la longitud del camino
            # de esta manera no se favorece inmediatamente a los cminos mas cortos
            deposito = Q
            for camino in caminos:
                for q in range(len(camino) - 1):
                    i = camino[q]
                    j = camino[q + 1]
                    sigma[i, j] += deposito
                    sigma[j, i] += deposito


        # GUARDAR EVOLUCION
        # guardamos la mejor longitud encontrada hasta el momento para poder analizar la convergencia
        historial.append(mejor_longitud)

    # devolvemos los resultados
    return mejor_camino, mejor_longitud, historial

# FUNCION PARA REALIZAR VARIAS CORRIDAS
def realizar_corridas(D,tipo_deposito,rho,Q,n_corridas=10,n_hormigas=17,alpha=1,beta=1,max_iter=500):
    # listas para guardar los resultados de cada corrida
    longitudes = []
    tiempos = []

    # guardamos tambien el mejor camino encontrado
    mejor_camino = None
    mejor_longitud = np.inf

    # historial de la mejor corrida
    mejor_historial = None

    # EJECUTAR VARIAS CORRIDAS

    for corrida in range(n_corridas):

        # utilizamos una semilla diferente en cada corrida
        seed = corrida

        # comenzamos a medir el tiempo
        inicio = time.perf_counter()

        # ejecutar el sistema de hormigas
        camino, longitud, historial = sistema_hormigas(D,n_hormigas=n_hormigas,alpha=alpha,beta=beta,rho=rho,Q=Q,max_iter=max_iter,tipo_deposito=tipo_deposito,seed=seed)

        # finalizar medicion del tiempo
        fin = time.perf_counter()

        tiempo = fin - inicio

        # guardar resultados
        longitudes.append(longitud)
        tiempos.append(tiempo)

        # actualizar mejor solucion de todas las corridas
        if longitud < mejor_longitud:
            mejor_longitud = longitud
            mejor_camino = camino.copy()
            mejor_historial = historial.copy()


    # CALCULAR ESTADISTICAS
    longitud_media = np.mean(longitudes)
    longitud_std = np.std(longitudes)
    tiempo_medio = np.mean(tiempos)
    tiempo_std = np.std(tiempos)

    # devolver resultados
    return {
        "tipo_deposito": tipo_deposito,
        "rho": rho,
        "Q": Q,
        "longitud_media": longitud_media,
        "longitud_std": longitud_std,
        "mejor_longitud": mejor_longitud,
        "tiempo_medio": tiempo_medio,
        "tiempo_std": tiempo_std,
        "mejor_camino": mejor_camino,
        "mejor_historial": mejor_historial
    }


# EXPERIMENTO 1: EFECTO DE RHO

# diferentes tasas de evaporacion que queremos comparar
valores_rho = [0.1, 0.3, 0.5, 0.7, 0.9]

# cantidad de feromona utilizada
Q = 1

# cantidad de corridas para cada configuracion
n_corridas = 10

# cantidad maxima de iteraciones
max_iter = 500

resultados_rho = []

print("\n")
print("============================================================")
print("EFECTO DE LA TASA DE EVAPORACION")
print("============================================================")

for rho in valores_rho:
    print("\nProbando rho =", rho)
    resultado = realizar_corridas(D,tipo_deposito="global",rho=rho,Q=Q,n_corridas=n_corridas,max_iter=max_iter)
    resultados_rho.append(resultado)

# convertir resultados a DataFrame
tabla_rho = pd.DataFrame(resultados_rho)
print("\nRESULTADOS PARA RHO")
print(tabla_rho[["tipo_deposito","rho","Q","longitud_media","longitud_std","mejor_longitud","tiempo_medio"]])

#COMPARAR CANTIDAD DE FEROMONA
# diferentes cantidades de feromona
valores_Q = [0.1, 0.5, 1, 2, 5]
# mantenemos rho fijo para analizar solamente
# el efecto de Q
rho = 0.5

resultados_Q = []

print("\n")
print("============================================================")
print("EFECTO DE LA CANTIDAD DE FEROMONA")
print("============================================================")

for Q in valores_Q:
    print("\nProbando Q =", Q)
    resultado = realizar_corridas(D,tipo_deposito="global",rho=rho,Q=Q,n_corridas=n_corridas,max_iter=max_iter)
    resultados_Q.append(resultado)

# convertir resultados a DataFrame
tabla_Q = pd.DataFrame(resultados_Q)

print("\nRESULTADOS PARA Q")
print(tabla_Q[["tipo_deposito","rho","Q","longitud_media","longitud_std","mejor_longitud","tiempo_medio"]])

# COMPARAR GLOBAL, LOCAL Y UNIFORME
# mantenemos rho y Q fijos para comparar solamente el tipo de deposito
rho = 0.5
Q = 1

tipos_deposito = ["global","local","uniforme"]

resultados_deposito = []
print("\n")
print("============================================================")
print("COMPARACION DE TIPOS DE DEPOSITO")
print("============================================================")

for tipo in tipos_deposito:
    print("\nProbando deposito:", tipo)
    resultado = realizar_corridas(D,tipo_deposito=tipo,rho=rho,Q=Q,n_corridas=n_corridas,max_iter=max_iter)
    resultados_deposito.append(resultado)

# convertir resultados a DataFrame
tabla_deposito = pd.DataFrame(resultados_deposito)

print("\nRESULTADOS PARA LOS TIPOS DE DEPOSITO")
print(tabla_deposito[["tipo_deposito","rho","Q","longitud_media","longitud_std","mejor_longitud","tiempo_medio"]])

# MEJOR RESULTADO GENERAL
# combinamos los resultados de los experimentos para poder consultar las configuraciones evaluadas
todos_resultados = pd.concat( [tabla_rho,tabla_Q,tabla_deposito],ignore_index=True)

# MOSTRAR EL MEJOR CAMINO DEL EXPERIMENTO DE DEPOSITO
print("\n")
print("============================================================")
print("MEJORES CAMINOS PARA LOS TIPOS DE DEPOSITO")
print("============================================================")

for resultado in resultados_deposito:
    print("\nDeposito:", resultado["tipo_deposito"])
    print("Mejor camino:")
    print( [int(ciudad) for ciudad in resultado["mejor_camino"]])
    print("Mejor longitud:")
    print(resultado["mejor_longitud"])
    print("Tiempo medio:")
    print(resultado["tiempo_medio"], "segundos")


#CONCLUSIONES
#Para el rho(EVAPORACION): Se observa que rho=0.1 obtiene el mejor comportamiento promedio, con una longitud media de 2085 y sin variabilidad entre las corridas.
#A medida que aumenta rho, aumenta también la longitud media de los caminos. Esto indica que una evaporación demasiado alta perjudica el desempeño del algoritmo.
#Esto tiene sentido porque una tasa de evaporación alta hace que la feromona desaparezca rápidamente. 
#Además, para rho=0.9 aparece un desvío estándar muy elevado (34.64). Esto indica que los resultados son mucho más variables entre las distintas ejecuciones.
#En cambio, con rho=0.1, la longitud media coincide con la mejor longitud encontrada:
#por lo que las corridas fueron extremadamente consistentes.

#Para el Q (Cantidad de Feromona): Acá no aparece una relación tan marcada como con rho.
#Todos los valores de Q lograron encontrar alguna vez el camino de longitud 2085: Lmin=2085
#Por lo tanto, la cantidad de feromona no impidió encontrar la mejor solución en ninguna de las configuraciones.
#Sin embargo, si observamos la longitud media y el desvío estándar, Q=1 presenta el mejor comportamiento: L=2090.7 con variacion=6.07
#Mientras que, por ejemplo, Q=0.5 tiene una longitud media de 2100.8 y un desvío de 20.59.
#Esto indica que Q=1 produjo soluciones más cercanas al mejor camino y más consistentes entre las diferentes corridas.
#Es importante notar que no podemos concluir que aumentar Q siempre empeore el algoritmo. Los valores probados no muestran una tendencia monotónica clara.
#Conclusión: el efecto de Q es menos significativo que el de rho en estos resultados. Dentro de los valores probados, Q=1 presenta el mejor equilibrio entre calidad y estabilidad.

#COMPARACION TIPOS DE DEPOSITOS
#Acá la diferencia más importante aparece entre el depósito local y los otros dos.
#El depósito global obtuvo: L_media=2090.7
#mientras que el uniforme obtuvo: L_media=2091.6
#Son resultados prácticamente equivalentes.
#En cambio, el depósito local obtuvo: L_media=2159.6 que es considerablemente peor.
#También se observa que el mejor camino encontrado por global y uniforme tiene longitud: L_min=2085
#mientras que el mejor encontrado por local fue: L_min=2143
#Por lo tanto, el depósito local fue claramente el menos efectivo en esta experiencia.

#En cuanto al tiempo, las diferencias son pequeñas:
#- Global: 3.73 s
#- Local: 3.83 s
#- Uniforme: 3.70 s
#Por lo tanto, el tiempo de ejecución no parece ser el factor determinante para explicar la diferencia de rendimiento. 
# #La principal diferencia está en la calidad de las soluciones obtenidas.

#COMPARACION DE LOS MEJORES CAMINOS
#Global y uniforme obtienen el mismo camino con longitud 2085, mientras que local encontró uno distinto con longitud 2143
