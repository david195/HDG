El algoritmo que se implementa es el de HDG (histogramas de divergencia de gradientes).
Este programa recibe los siguientes parámetros:

Dirección de la imagen, la cual deverá ser cuadrada.
Número de regiones.
Número de bins.
Tipo de normalización
    0 : Sin normalización.
    1 : normalización de minimos y maximos.
    2 : normalización L2.

Como salida se imprime en consola el vector característico obtenido y se muestran en ventana 
la representación del histograma de cada region.

Compilación del script.
	> cmake .
	> make

Ejecución del script:
	>  . /main  img.jpg 8 64 0


 
