import pandas as pd
import random
from entrenar_neurona import entrenar_neurona 
ubi_50="C:/Users/conra/OneDrive/Desktop/Facu Conrado/CUARTO AÑO/Inteligencia Computacional/Tema 1 Redes neuronales/OR_50_trn.csv"
df_50=pd.read_csv(ubi_50)
print(df_50.head(3))
ubi_90="C:/Users/conra/OneDrive/Desktop/Facu Conrado/CUARTO AÑO/Inteligencia Computacional/Tema 1 Redes neuronales/OR_90_trn.csv"
df_90=pd.read_csv(ubi_90)
print(df_90.head(3))
n = df_50.shape[1]   # tiene dimensión 3 por x1 y x2 y bias
w = [0.0] * n
for i in range(len(w)):
    # use uniform para float en el rango [-0.5, 0.5]
    w[i] = random.uniform(-0.5, 0.5)
    print(w[i])
epocas=10 #nro max de epocas (preg si hay alguna forma de hallar uno "optimo" para no ponerlo a ojo)
nformula=0.5
w50_historial,w50_final,errores_historial_50 = entrenar_neurona(w, epocas, df_50, nformula) 
print("cantidad de epocas realizadas:", len(w50_historial))
print("errores por epoca:", errores_historial_50)
w=w50_final
#como en el ej1 ahora ya estoy en cond de comparar
ubi_tst_50="C:/Users/conra/OneDrive/Desktop/Facu Conrado/CUARTO AÑO/Inteligencia Computacional/Tema 1 Redes neuronales/OR_50_tst.csv"
df_tst_50=pd.read_csv(ubi_tst_50)
ubi_tst_90="C:/Users/conra/OneDrive/Desktop/Facu Conrado/CUARTO AÑO/Inteligencia Computacional/Tema 1 Redes neuronales/OR_90_tst.csv"
df_tst_90=pd.read_csv(ubi_tst_90)
#aciertos para el de 50
aciertos_50=0
for i in range(len(df_tst_50)):
    x1=df_tst_50.iloc[i,0]
    x2=df_tst_50.iloc[i,1]
    yd=df_tst_50.iloc[i,2]
    x=[-1,x1,x2]
    v=0
    for j in range(len(w)): #aca ya estoy en condicion de hacaer la prediccion con los w que ya encontre y los x del test
        v=v+w[j]*x[j] 
        if v>=0:
            y=1
        else:
            y=-1
    if y==yd:
        aciertos_50=aciertos_50+1
porc_aciert_50=(aciertos_50*100)/(len(df_tst_50))
print("para el de desviacion 50% hubo: ",aciertos_50,"aciertos")
print("el porcentaje de aciertos es d: ",porc_aciert_50)

#ahora para el de 90
aciertos_90=0
for i in range(len(df_tst_90)):
    x3=df_tst_90.iloc[i,0]
    x4=df_tst_90.iloc[i,1]
    yd2=df_tst_90.iloc[i,2]
    x=[-1,x1,x2]
    v=0
    for j in range(len(w)):
        v=v+w[j]*x[j]
        if v>=0:
            y=1
        else:
            y=-1
    if y==yd2:
        aciertos_90=aciertos_90+1
porc_aciert_90=(aciertos_90*100)/(len(df_tst_90))
print("para el de desviacion 90% hubo: ",aciertos_90,"aciertos")
print("el porcentaje de aciertos es de: ",porc_aciert_90)

#los resultados que tengo son: para el de desviacion 50% hubo:  199 aciertos
#el porcentaje de aciertos es d:  100.0
#para el de desviacion 90% hubo:  149 aciertos
#el porcentaje de aciertos es de:  74.87437185929649

#entiendo que esta diferencia viene de que, al agregarle una desviacion a los datos le estamos agregando ruido, en el caso
#del 50% el ruido no es lo suficientemente alto como para que el percepton no aprenda los patrones, es decir, que sigue pudiendo encontrar
#una recta que separe las clases, con una exactitud del 100%.
#Para el caso del 90% el ruido que se introduce ya hace que no se puedan separar claramente las clases con una recta, ademas,
#crea solapamientos, lo q puede generar que al generar la recta, la misma no pueda separar de manera correcta a las clases y por ello
#tenemos que, la exactitud cae al 75%.
