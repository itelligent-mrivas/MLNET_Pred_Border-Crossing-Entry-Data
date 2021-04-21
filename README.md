# MLNET_Pred_Border-Crossing-Entry-Data
Construcción de un Modelo de Regresión a partir de ML.NET para predecir el número de vehículos que cruzan la frontera de US. 
Para la construcción del mencionado Modelo, en primer lugar, se crea la clase de representación de observaciones a través de una tabla de datos, (extraída de la BBDD “Border Crossing/Entry Data” y manipulada para crear nuevas features), la cual es transformada en un IDataView y dividida a su vez en dos IDataView, (entrenamiento y testeo). 
Posteriormente se establecen dos pipeline, los cuales contienen, por un lado, las transformaciones de los datos y por el otro los Algoritmos de Entrenamiento de ML.NET seleccionados, Gam y FastTree, y se entrena el Modelo. 
Se concluye el programa evaluando el Modelo a través de métricas características de los Modelos de Regresión y seleccionado uno de los Modelos entrenados con los algoritmos seleccionados (FastTree).
