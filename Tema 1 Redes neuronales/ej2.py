from entrenar_neurona import entrenar_neurona 
import pandas as pd
import random
import matplotlib.pyplot as plt
import ej1
ubi="C:/Users/conra/OneDrive/Desktop/Facu Conrado/CUARTO AÑO/Inteligencia Computacional/Tema 1 Redes neuronales/OR_trn.csv"
df=pd.read_csv(ubi)
#entiendo que tengo que reutilizar los valores que obtuve en el entrenamiento del ej 1, preguntar si es asi
#siguiendo esa base y haciendo el import correspondiente puedo acceder a los datos obtenidos del ej 1
historial_wor=ej1.w_historial
ubi2="C:/Users/conra/OneDrive/Desktop/Facu Conrado/CUARTO AÑO/Inteligencia Computacional/Tema 1 Redes neuronales/XOR_trn.csv"
df2=pd.read_csv(ubi2)
epocas=10
#esto es igual q en el ej 1, porq tengo q entrenar para el xor ahora
n = df.shape[1]   # tiene dimensión 3 por x1 y x2 y bias
w = [0.0] * n
nformula=0.5
for i in range(len(w)):
    # use uniform para float en el rango [-0.5, 0.5]
    w[i] = random.uniform(-0.5, 0.5)
    print(w[i])
historial_wxor,wxor,errores_xor=entrenar_neurona(w,epocas,df2,nformula) #entreno con los datos el xor 

#PARA EL OR
for epoca in range(len(historial_wor)):
    w = historial_wor[epoca]
    plt.figure()
    # puntos fijos
    for i in range(len(ej1.df)):
        x1 = ej1.df.iloc[i, 0]
        x2 = ej1.df.iloc[i, 1]
        yd = ej1.df.iloc[i, 2]
        if yd == 1:
            plt.scatter(x1, x2, marker="o")
        else:
            plt.scatter(x1, x2, marker="x")
    # recta
    x1_recta = [-2, 2]
    x2_recta = []
    for x1 in x1_recta:
        x2 = (w[0] - w[1] * x1) / w[2]
        x2_recta.append(x2)
    plt.plot(x1_recta, x2_recta)
    # mismos ejes siempre
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("OR - epoca " + str(epoca + 1))
    plt.grid()
    plt.show()