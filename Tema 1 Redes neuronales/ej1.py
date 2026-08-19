import pandas as pd
import random
from entrenar_neurona import entrenar_neurona 

ubi="C:/Users/conra/OneDrive/Desktop/Facu Conrado/CUARTO AÑO/Inteligencia Computacional/Tema 1 Redes neuronales/OR_trn.csv"
df=pd.read_csv(ubi)
print(df.head())
ubtest="C:/Users/conra/OneDrive/Desktop/Facu Conrado/CUARTO AÑO/Inteligencia Computacional/Tema 1 Redes neuronales/OR_tst.csv"
df2=pd.read_csv(ubtest)
print(df2.head())
#sabemos que tenemos la salida en la tercera columna, es decir, que es, x1,x2,y
# tengo que inicializar los pesos que van a estar entre -0.5 y 0.5
# inicializo un vector de ceros con la longitud igual al número de características (x1,x2)
n = df.shape[1]   # tiene dimensión 3 por x1 y x2 y bias
w = [0.0] * n
for i in range(len(w)):
    # use uniform para float en el rango [-0.5, 0.5]
    w[i] = random.uniform(-0.5, 0.5)
    print(w[i])
#ahora ya arme el vector de pesos entonces puedo empezar a hacer el prod interno como especifico el profe
#print(df.iloc[:, 0])
#print(df.iloc[:, 1])
#print(df.iloc[:, 2])
epocas=10 #nro max de epocas (preg si hay alguna forma de hallar uno "optimo" para no ponerlo a ojo)
nformula=0.5
w_historial,w_final,errores_historial = entrenar_neurona(w, epocas, df, nformula) 
print("cantidad de epocas realizadas:", len(w_historial))
print("errores por epoca:", errores_historial)
w=w_final
#ahora ya puedo comparar con los datos de train 
aciertos=0
for i in range(len(df2)):
    x1=df2.iloc[i,0]
    x2=df2.iloc[i,1]
    yd=df2.iloc[i,2]
    x=[-1,x1,x2]
    v=0
    for j in range(len(w)): #aca ya estoy en condicion de hacaer la prediccion con los w que ya encontre y los x del test
        v=v+w[j]*x[j] 
    if v>=0:
        y=1
    else:
        y=-1
    if y==yd:
        aciertos=aciertos+1
porc_aciert=(aciertos*100)/len(df2)
print("hubo: ",aciertos,"aciertos")
print("el porcentaje de aciertos es d: ",porc_aciert)

#Preguntar: si se puede armar como funciones por ejemplo, el tema de entrenar hacerlo como
#def entrenar_neurona(....)


    