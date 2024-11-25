# Data Mining en Economía y Finanzas
#### *Segunda entrega*

## Corrida general del Workflow Baseline

Como mis exploraciónes no estaban llegando a buen puerto corté el sabado y retomé modelos que estuvieran circulando y que anduvieran bien.

Tomé de base el scipt de Alejandro Czernikier que ya tenía implementado el tema de incorporar muchos meses, lag1 y lag2 y lag de segundo orden (12 meses). 
Después modifiqué algunas cosas:
[Link al original](https://github.com/alejandro-czernikier/dmeyf2024/blob/main/src/workflows/918_workflow_base_f202108_1_sinmarzoabril_5modelos_tendencia2.r)


1. En CA_catastrophe_base agregé lineas a corregir robadas de Joaquin Tschopp
[Link al original](https://github.com/JoacoTschopp/dmeyf2024/blob/main/src/wf-etapas/1201_CA_reparar_dataset.r)

2. FErf_attributes_base cambié arbolitos a 25 y 16 hojas
para que aduviera más rápido en la TS undersamplin 0.02 en train y 0.2 en finaltrain

3. 30 iteraciones de la bayesiana y elegí los tres primeros modelos con 5 semillas
hubiera usado 20  pero me quedé sin tiempo

4. desde python con el archivo de los scores, hice un promedio de las semillas por modelo y enrtegué ese promedio.

Elegí para la entrega un buen score del promedio de los tres modelos. 

