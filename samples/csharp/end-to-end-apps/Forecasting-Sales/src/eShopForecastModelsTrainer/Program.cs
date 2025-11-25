using Microsoft.ML;
using System;

namespace eShopForecastModelsTrainer
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // Path CSV (pastikan sudah sesuai)
            string dataPath = "data/cosmetics_sales_data.csv";

            DailyTimeSeriesHelper.PerformTimeSeriesForecasting(mlContext, dataPath);
        }
    }
}
