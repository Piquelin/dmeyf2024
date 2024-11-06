# Data Mining en Economía y Finanzas
#### *Primera entrega*

## Resumen
Para la entrega que resultó la mejor en kaggle fuimos haciendo una exploración de hiperparametro arrancando de la base que daba la cátedra que era buena.
La que mejor resultado nos dió fue la exploración de Quantización donde busamos los hiperparametros de cantidad de bines entre 3 y 7 (porque el default era 4).
Obtubiomos buenos resultados con la exploraciónde baggin pero cuando quisimos combinarlos aparecienron errores en la corrida final de lgbm y dejamos el experimento para más adelante.

Hicimos ensayos de uso de clusterización pero no hubo una mejora en los resultados, por lo que pensamos dedicarle más timepo en los ensayos colaborativos.

## Paso a paso
En todo el proceso aprovechamos el pipeline brindado por la cátedra y para cada nuevo experimento cambiamos los nombre de manera que se mantuviera la ivisión ordenada por carpetas.
0) Para la creaciónde la clase ternaria desde el archivo base utulizamos 00_clase_ternaria.py que crea la clase para los meses 01 a 04, dejando nulos los dos ultimos meses. Por consejo de al cátedra se sacó acá las dos columnas de prestamos personales que presenaban concept drifting.
1) En la parte de preprocesamiento 01_723_preprocesamiento_entrega.r se toma el erchivo brindado por la cátedra y se modificaalgunas pequeñas cosas.
  - se hace un undersamplig de 0.25 en el trai para la clase mayoritaria.
  - se eliminan los controles para limitar memoria porque tienen problemas con windows y como mi flujo de trabajo implica trabajar en local en másquinas con windows tenía problemas
  - se agrega a las columnas para corregir ccajas_depositos del mes 05
  - eliminamos la variable de fecha de cierre máx de las tarjetas "vm_fultimo_cierre"
2) para la optimización bayesiana, 02_724_lightgbm_binaria_BO.r, agregamos estos parametros:
  -   use_quantized_grad = TRUE,
  -   (num_grad_quant_bins =  4,) este lo comentamos porque después lo incluimos en la búsqueda de hiperparametros 
  -   quant_train_renew_leaf = TRUE
  dentro de los hiperparametros a optimizar agregamos este:
  -   makeIntegerParam("num_grad_quant_bins", lower = 3L, upper = 7L)
3) finalmente corrimos el 03_745_lightgbm_final.r cambiando solo la semilla

Para la selección del punto de corte hicimos un análisis del comportamiento de los envíos y elegimos un envío del segundo modelo.
![comparativa_envios](https://github.com/Piquelin/dmeyf2024/blob/main/varios/Primera_entrega_kaggle/comparacion_modelos_y_semillas.png?raw=true)

