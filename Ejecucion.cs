using System;

using System.IO;
using Microsoft.ML;

using System.Linq;

using Microsoft.ML.Trainers;
using static Microsoft.ML.DataOperationsCatalog;

namespace MLNET_Pred_Border_Crossing_Entry_Data
{
    class Ejecucion
    {
        //Rutas de acceso de dataset y Modelos
        static readonly string _DataPath = @"./DATOS/Border_Crossing_Entry_Data_trans.csv";
        static readonly string _salida_trainDataPath = @"./DATOS/trainData.csv";
        static readonly string _salida_testDataPath = @"./DATOS/testData.csv";
        static readonly string _salida_transformationData = @"./DATOS/transformationData.csv";
        static readonly string _salida_modelPath = @"./DATOS/model.zip";
        static void Main(string[] args)
        {
            //###############################################################
            //INICIALIZACIÓN DEL PROCESO
            //###############################################################

            //Inicialización de mlContext; utilización del seed para replicidad
            MLContext mlContext = new MLContext(seed: 1);

            //Definición de las clases de los datos de entrada: 
            //  -Clase Observaciones: BorderCrossObservation

            //Carga de datos
            IDataView originalFullData = 
                mlContext.Data.LoadFromTextFile<BorderCrossObservation>(
                    _DataPath, 
                    separatorChar: ';', 
                    hasHeader: true);


            //###############################################################
            //CONSTRUYE EL CONJUNTO DE DATOS (DATASET)
            //###############################################################

            //División del IDataView originalFullData:
            //  -entrenamiento (trainingDataView): 80% 
            //  -testeo (testDataView): 20%

            //Selección de porcentaje para el test
            double testFraction = 0.2;
            
            //Aplicacón de la División
            TrainTestData Split_trainTestData = mlContext.Data.TrainTestSplit(originalFullData, 
                testFraction: testFraction, seed: 1);

            //IDataView resultantes
            IDataView trainingDataView = Split_trainTestData.TrainSet;
            IDataView testDataView = Split_trainTestData.TestSet;

            //Guardar IDataView trainingDataView para una posible viasualización (extensión csv)
            using (var fileStream = File.Create(_salida_trainDataPath))
            {
                mlContext.Data.SaveAsText(trainingDataView, fileStream, separatorChar: ';', headerRow: true, schema: true);
            }

            //Guardar IDataView testDataView para una posible viasualización (extensión csv)
            using (var fileStream = File.Create(_salida_testDataPath))
            {
                mlContext.Data.SaveAsText(testDataView, fileStream, separatorChar: ';', headerRow: true, schema: true);
            }


            //###############################################################
            //SELECCIÓN DE VARIABLES
            //###############################################################

            //Suprimimos del esquema IDataView lo que no seleccionemos como features
            var listfeatureColumnNames = trainingDataView.Schema.AsQueryable()
                .Select(column => column.Name)
                .Where(name => name != "Label" && //atributo de salida 
                name != "Port_Name" && //solo existe un valor
                name != "State" && //un valor
                name != "Port_Code" && //un valor
                name != "Border" && //un valor
                name != "Longitud" && //un valor
                name != "Latitud" && //un valor
                name != "Mes" && //transformar
                name != "Year" && //transformar
                name != "Measure").ToList(); //transformar

            //Añadimos las Transformaciones de los atributos suprimidos anteriormente             
            listfeatureColumnNames.Add("MesInd");
            listfeatureColumnNames.Add("YearInd");
            listfeatureColumnNames.Add("MeasureInd");

            //Conversión a array para su posterior utlización
            string[] featureColumnNames = listfeatureColumnNames.ToArray();


            //###############################################################
            //TRANFORMACIÓN DE LOS DATOS DEL MODELO --> pipeline
            //###############################################################            

            //Indicadoras
            IEstimator<ITransformer> pipeline = mlContext.Transforms.Categorical.OneHotEncoding(
                outputColumnName: "MesInd", inputColumnName: "Mes")
            //Indicadoras
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                outputColumnName: "YearInd", inputColumnName: "Year"))
            //Indicadoras
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                outputColumnName: "MeasureInd", inputColumnName: "Measure"))
            //Concatena
            .Append(mlContext.Transforms.Concatenate(
                "Features", featureColumnNames))
            //Surpime del IDataView
            .Append(mlContext.Transforms.DropColumns(
                new string[] { "Mes", "Year", "Measure" }))
            //Normalizado del atributo de salida
            .Append(mlContext.Transforms.NormalizeMeanVariance(
                inputColumnName: "Label", outputColumnName: "LabelNormalized"));

            //Guardar datos transformedData        
            IDataView transformedData =
                pipeline.Fit(trainingDataView).Transform(trainingDataView);
            using (var fileStream = File.Create(_salida_transformationData))
            {
                mlContext.Data.SaveAsText(transformedData, fileStream, separatorChar: ';', headerRow: true, schema: true);
            }


            //###############################################################
            //SELECCIÓN DE ALGORITMOS DE ENTRENAMIENTO --> trainingPipeline
            //###############################################################

            //***************************************************************
            //1. GAM (Generalized Additive Models)
            //***************************************************************            

            var trainer_gam = mlContext.Regression.Trainers
                .Gam(labelColumnName: "LabelNormalized",
                featureColumnName: "Features",
                learningRate: 0.02,
                numberOfIterations: 2100);

            //Se añade el Algoritmo al pipeline de transformación de datos
            IEstimator<ITransformer> trainingPipeline_gam = pipeline.Append(trainer_gam);


            //***************************************************************
            //2. GBA (Gradient Boosting Algorithm)
            //***************************************************************           

            var trainer_boost = mlContext.Regression.Trainers                
                .FastTree(labelColumnName: "LabelNormalized",
                featureColumnName: "Features",
                numberOfLeaves: 20,
                numberOfTrees: 100,
                minimumExampleCountPerLeaf: 10,
                learningRate: 0.2);

            //Se añade el Algoritmo al pipeline de transformación de datos            
            IEstimator<ITransformer> trainingPipeline_boost = pipeline.Append(trainer_boost);


            //###############################################################
            //ENTRENAMIENTO DE LOS MODELOS
            //###############################################################

            Console.WriteLine($"\n************************************************************");
            Console.WriteLine($"* Entrenamiento del Modelo calculado con el Algoritmo GAM   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            var watch_gam = System.Diagnostics.Stopwatch.StartNew();
            var model_gam = trainingPipeline_gam.Fit(trainingDataView);
            watch_gam.Stop();
            var elapseds_gam = watch_gam.ElapsedMilliseconds*0.001;
            Console.WriteLine($"El entrenamiento GAM ha tardado: {elapseds_gam:#.##} s\n");

            Console.WriteLine($"\n************************************************************");
            Console.WriteLine($"* Entrenamiento del Modelo calculado con el Algoritmo GBA   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            var watch_boost = System.Diagnostics.Stopwatch.StartNew();
            var model_boost = trainingPipeline_boost.Fit(trainingDataView);
            watch_boost.Stop();
            var elapseds_boost = watch_boost.ElapsedMilliseconds * 0.001;
            Console.WriteLine($"El entrenamiento GBA ha tardado: {elapseds_boost:#.##} s\n");


            //###############################################################
            //EVALUACIÓN DE LOS MODELOS
            //###############################################################

            //Transformación del IDataView testDataView a paritr de ambos modelos
            var predictions_gam = model_gam.Transform(testDataView);
            var predictions_boost = model_boost.Transform(testDataView);

            //Calculo de las métricas de cada Modelo
            var metrics_gam = mlContext.Regression
                .Evaluate(data: predictions_gam, labelColumnName: "LabelNormalized", scoreColumnName: "Score");
            var metrics_boost = mlContext.Regression
                .Evaluate(data: predictions_boost, labelColumnName: "LabelNormalized", scoreColumnName: "Score");            

            //Muestra las métricas GAM
            Console.WriteLine($"\n************************************************************");
            Console.WriteLine($"* Métricas para el Modelo calculado con el Algoritmo GAM      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       GAM RSquared Score:      {metrics_gam.RSquared:0.##}");
            Console.WriteLine($"*       GAM Root Mean Squared Error Score:      {metrics_gam.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*       GAM MAE Score:      {metrics_gam.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*       GAM MSE Score:      {metrics_gam.MeanSquaredError:#.##}\n");

            //Muestra las métricas GBA
            Console.WriteLine($"\n************************************************************");
            Console.WriteLine($"* Métricas para el Modelo calculado con el Algoritmo GBA      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       GBA RSquared Score:      {metrics_boost.RSquared:0.##}");
            Console.WriteLine($"*       GBA Root Mean Squared Error Score:      {metrics_boost.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*       GBA MAE Score:      {metrics_boost.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*       GBA MSE Score:      {metrics_boost.MeanSquaredError:#.##}\n");


            //###############################################################
            //SELECCIÓN DEL MEJOR MODELO
            //###############################################################

            //Guardamos el Modelo para su posterior consumo
            mlContext.Model.Save(model_boost, trainingDataView.Schema, _salida_modelPath);


        }
    }
}
