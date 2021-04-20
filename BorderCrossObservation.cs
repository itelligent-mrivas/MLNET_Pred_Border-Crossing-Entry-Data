using System;
using System.Collections.Generic;
using System.Text;

using Microsoft.ML.Data;

namespace MLNET_Pred_Border_Crossing_Entry_Data
{
    class BorderCrossObservation
    {
        //#####################################################################################################################################
        //WEB DATOS: https://www.kaggle.com/akhilv11/border-crossing-entry-data
        //BBDD OFICIAL DATOS: https://www.bts.gov/browse-statistical-products-and-data/border-crossing-data/border-crossingentry-data
        //#####################################################################################################################################

        //Definición del nombre y del dominio de los atributos en relación con las columnas de la tabla de obs
        //No es necesario cargar todos los atributos de la tabla, solo los que se vayan a utilizar

        [LoadColumn(0)]
        public string Port_Name;//El Paso

        [LoadColumn(1)]
        public string State;//Texas

        [LoadColumn(2)]
        public string Port_Code;//2402

        [LoadColumn(3)]
        public string Border;//US-Mexico Border

        [LoadColumn(4)]
        public string Longitud;//-106.48639

        [LoadColumn(5)]
        public string Latitud;//31.758610000000004        
       
        [LoadColumn(6)]
        public string Mes;//1 (300), 2 (300), 3 (300), 4 (300), 5 (299), 6 (300), 7 (300), 8 (300), 9 (299), 10 (299), 
                          //11 (300), 12 (299)
                          //--> 300 obs = 25 años x 12 Measure; Faltan: 4 obs del valor "Train Passenger"

        [LoadColumn(7)]
        public string Year;//1996 (144), 1997 (144), 1998 (144), 1999 (144), 2000 (144), 2002 (144), 2003 (144), 2004 (144), 
                           //2005 (144), 2006 (144), 2007 (144), 2008 (144), 2009 (144), 2010 (144), 2011 (144), 2012 (144), 
                           //2013 (144), 2014 (144), 2015 (144), 2016 (144), 2017 (144), 2018 (144), 2019 (143), 2020 (141)
                           //--> 12 Measure x 12 Mes = 144 obs; Falta: 1 de 2019, 3 de 2020; (valor "Train Passenger")       

        [LoadColumn(8)]
        public string Measure;//Truck Containers Full (300), Truck Containers Empty (300), Trucks (300), 
                              //Personal Vehicles (300), Personal Vehicle Passengers (300), 
                              //Trains (300), Train Passengers (296), 
                              //Rail Containers Empty (contenedores que legan a un puerto determinado, 300), Rail Containers Full (300),
                              //Buses (puede (o no) llevar pasajeros; 300), Bus Passengers (300),
                              //Pedestrians (300)
                              //-->25 años x 12 mes = 300 obs

        [LoadColumn(9)]
        public float Label;//cambia el nombre del atributo original en la tabla (Value) por el de "Label";
                           //maximo = 4447374 (solo se da 1 vez, Personal Vehicle Passengers); 
                           //minimo = 0; media = 333312
    }
}
