using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TimeSeries;
using ScottPlot; // ScottPlot untuk visualisasi
using System;
using System.IO;
using System.Linq;

namespace eShopForecastModelsTrainer
{
    public class DailyData
    {
        [LoadColumn(4)] // kolom ke-5 adalah Amount
        public float Amount { get; set; }
    }

    public class ForecastResult
    {
        [VectorType(7)]
        public float[] ForecastedAmount { get; set; } = new float[7];
    }

    public static class DailyTimeSeriesHelper
    {
        public static void PerformTimeSeriesForecasting(MLContext mlContext, string dataPath)
        {
            if (!File.Exists(dataPath))
            {
                Console.WriteLine($"File CSV tidak ditemukan: {dataPath}");
                return;
            }

            // Load CSV
            var data = mlContext.Data.LoadFromTextFile<DailyData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',');

            // SSA parameters
            int horizon = 7;
            int windowSize = 7;
            int seriesLength = 30;
            int trainSize = 30;

            // Build SSA Forecasting estimator
            var ssaEstimator = mlContext.Forecasting.ForecastBySsa(
                outputColumnName: nameof(ForecastResult.ForecastedAmount),
                inputColumnName: nameof(DailyData.Amount),
                windowSize: windowSize,
                seriesLength: seriesLength,
                trainSize: trainSize,
                horizon: horizon
            );

            // Train model
            var model = ssaEstimator.Fit(data);

            // Transform data untuk prediksi
            var forecastEngine = model.Transform(data);
            var forecastColumn = forecastEngine.GetColumn<float[]>(nameof(ForecastResult.ForecastedAmount));

            Console.WriteLine("Forecast 7 hari ke depan:");
            float[] forecastValues = Array.Empty<float>();
            int day = 1;

            foreach (var f in forecastColumn)
            {
                forecastValues = f;
                foreach (var amount in f)
                {
                    Console.WriteLine($"Hari {day}: {amount:0.00}");
                    day++;
                }
                break; // hanya 1 row prediksi
            }


            // Visualisasi dengan ScottPlot
            if (forecastValues != null)
            {
                double[] days = Enumerable.Range(1, forecastValues.Length).Select(x => (double)x).ToArray();
                double[] amounts = forecastValues.Select(x => (double)x).ToArray();

                var plt = new ScottPlot.Plot(600, 400);
                plt.AddScatter(days, amounts);
                plt.Title("Forecast 7 Hari ke Depan");
                plt.XLabel("Hari");
                plt.YLabel("Amount");
                plt.SaveFig("forecast.png");

                Console.WriteLine("Grafik disimpan sebagai forecast.png");
            }

            Console.WriteLine("Selesai.");
        }
    }
}
