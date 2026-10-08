import numpy as np
# funcion objetivo del ejercicio 1 de la GTP6
# el objetivo es encontrar el valor de x que minimiza f(x)
def funcion(x):
    return -x * np.sin(np.sqrt(np.abs(x)))

# parametros del problema
# limites del espacio de busqueda
xmin = -512
xmax = 512
# cantidad de particulas del enjambre
N = 30
# constantes de aceleracion
# c1 controla la influencia de la experiencia personal
# c2 controla la influencia de la experiencia social
c1 = 2
c2 = 2
# cantidad maxima de iteraciones del algoritmo
max_iter = 1000

# inicializacion

# generador de numeros aleatorios se fija la semilla para poder repetir el experimento
# obteniendo siempre la misma secuencia de numeros aleatorios
rng = np.random.default_rng(0)

# posiciones iniciales de las particulas cada particula comienza en una posicion aleatoria
# dentro del intervalo [xmin, xmax]
x = rng.uniform(xmin, xmax, N)

# velocidades iniciales de las particulas todas comienzan con velocidad cero
v = np.zeros(N)

# mejor historico de cada particula

# mejor posicion personal de cada particula inicialmente es la propia posicion inicial
y = x.copy()

# valor de la funcion en la mejor posicion personal fy[k] representa f(y[k])
fy = funcion(y)

# mejor posicion global

# busco el indice de la particula que tiene el menor valor de la funcion
indice_mejor = np.argmin(fy)

# mejor posicion encontrada por todo el enjambre representa la mejor posicion global y_hat
y_glob = y[indice_mejor]

# valor de la funcion en la mejor posicion global
f_y_glob = fy[indice_mejor]

# velocidad de convergencia

# usamos el mismo valor de referencia q en el algoritmo genetico
# para poder comparar cuantos pasos necesita cada algoritmo
umbral_convergencia = -418.9

# guardamos la primera iteracion en la q se alcanza el umbral
iteracion_convergencia = None

# evolucion

# lista para guardar el mejor valor global encontrado en cada iteracion
# permite analizar la convergencia del algoritmo
historial = []

for t in range(max_iter):
    # actualizar mejores historicos
    # evaluo la funcion en la posicion actual de cada particula
    fx = funcion(x)

    # comparo la posicion actual de cada particula con su mejor posicion personal encontrada hasta ahora
    # como estamos minimizando, una posicion es mejor cuando produce un valor menor de la funcion
    mejores = fx < fy

    # para las particulas que mejoraron, actualizo su mejor posicion personal
    y[mejores] = x[mejores]

    # actualizo tambien el valor de la funcion asociado a su mejor posicion personal
    fy[mejores] = fx[mejores]

    # actualizar mejor global
    # busco cual es actualmente la mejor posicion personal de todo el enjambre
    indice_mejor = np.argmin(fy)

    # si la mejor posicion actual es mejor que la mejor posicion global que tenia guardada,
    # actualizo la mejor posicion global
    if fy[indice_mejor] < f_y_glob:
        y_glob = y[indice_mejor]
        f_y_glob = fy[indice_mejor]


    # guardo el mejor valor global encontrado en esta iteracion para analizar la evolucion
    historial.append(f_y_glob)

    # velocidad de convergencia

    # verificamos si el mejor valor encontrado ya alcanzo
    # el mismo nivel de calidad usado en el algoritmo genetico
    if (iteracion_convergencia is None and f_y_glob <= umbral_convergencia): #usamos el is none xq solo nos interesa guardar la primera veez q alcanzamos el umbral
        #digamos q si todavía no registramos la convergencia y el mejor valor encontrado ya alcanzo el umbral, guardamos esta iteracion
        # guardamos la cantidad de iteraciones necesarias
        iteracion_convergencia = t + 1

    # generar numeros aleatorios


    # genero un numero aleatorio entre 0 y 1 para cada particula
    # r1 controla de forma aleatoria la influencia  de la experiencia personal
    r1 = rng.uniform(0, 1, N)
    # r2 controla de forma aleatoria la influencia de la experiencia social
    r2 = rng.uniform(0, 1, N)
 
    # ctualizar velocidad
    # ecuacion de actualizacion de la velocidad:
    # v(t+1) = v(t)+ c1*r1*(y - x)+ c2*r2*(y_glob - x)
    # v -> velocidad anterior
    # y - x -> experiencia personal
    # y_glob - x -> experiencia social
    # c1 y c2  -> constantes de aceleracion
    # r1 y r2  -> componentes aleatorias
    # la particula combina su movimiento anterior, su mejor experiencia y la mejor experiencia
    # encontrada por todo el enjambre

    v = (v+ c1 * r1 * (y - x)+ c2 * r2 * (y_glob - x))
    # actualizar posicion
    # actualizo la posicion de cada particula utilizando su nueva velocidad
    # x(t+1) = x(t) + v(t+1)
    x = x + v
    #mantener las particulas dentro del dominio
    # las particulas deben permanecer dentro del intervalo definido para el problema
    # si una particula sale del intervalo [-512, 512], se fuerza su posicion al limite correspondiente
    x = np.clip(x, xmin, xmax)

# resultado
# muestro la mejor posicion encontrada por todo el enjambre
print("Mejor posición encontrada:", y_glob)
# muestro el valor minimo de la funcion
# correspondiente a esa posicion
print("Valor de la función:", f_y_glob)

# velocidad de convergencia

print()
print("==============================")
print("velocidad de convergencia")

# mostramos cuantas iteraciones necesito pso para alcanzar
# el mismo nivel de calidad q usamos para comparar con el algoritmo genetico
print("umbral de convergencia:", umbral_convergencia)

if iteracion_convergencia is not None:
    print(
        "iteraciones necesarias:",
        iteracion_convergencia
    )
else:
    print("el umbral de convergencia no fue alcanzado")