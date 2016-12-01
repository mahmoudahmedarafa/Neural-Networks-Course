using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Neural_Task_4
{
    class RBF
    {
        List<Matrix<double>>[] training_set, testing_set, clusters;   //1-based arrays
        int epochs;
        double eta;
        int[,] confusion_matrix;
        int num_of_clusters;
        Matrix<double>[] centroids;     //1-based array
        double[] variances, phi;
        double[,] weights, temp_weights;
        DataGridView dataGridView;

        public RBF(double eta, int epochs, int num_of_clusters, ref DataGridView dataGridView)
        {
            this.eta = eta;
            this.epochs = epochs;
            this.num_of_clusters = num_of_clusters;
            this.dataGridView = dataGridView;
            training_set = new List<Matrix<double>>[4];
            testing_set = new List<Matrix<double>>[4];
            clusters = new List<Matrix<double>>[num_of_clusters + 1];
            for (int i = 0; i < 4; i++)
            {
                training_set[i] = new List<Matrix<double>>();
                testing_set[i] = new List<Matrix<double>>();
            }
            for (int i = 0; i <= num_of_clusters; i++)
            {
                clusters[i] = new List<Matrix<double>>();
            }
            confusion_matrix = new int[5, 5];
            centroids = new Matrix<double>[num_of_clusters + 1];
            variances = new double[num_of_clusters + 1];
            phi = new double[num_of_clusters + 1];
            weights = new double[4, num_of_clusters + 1];

            temp_weights = new double[4, num_of_clusters + 1];
        }

        public void displayConfusionMatrix()
        {
            dataGridView.Rows.Clear();
            for (int i = 1; i < 5; i++)
            {
                var row = new DataGridViewRow();
                for (int j = 1; j < 5; j++)
                {
                    row.Cells.Add(new DataGridViewTextBoxCell()
                    {
                        Value = confusion_matrix[i, j]
                    });
                }
                dataGridView.Rows.Add(row);
            }
        }

        public void prepareConfusionMatrix()
        {
            for (int i = 1; i < 5; i++)
            {
                for (int j = 1; j < 4; j++)
                {
                    confusion_matrix[i, 4] += confusion_matrix[i, j];
                }
            }

            for (int i = 1; i < 5; i++)
            {
                for (int j = 1; j < 4; j++)
                {
                    confusion_matrix[4, i] += confusion_matrix[j, i];
                }
            }
        }

        public string calculateOverallAccuracy()
        {
            int sumDiagonal = 0;
            for (int i = 1; i <= 3; i++)
            {
                sumDiagonal += confusion_matrix[i, i];
            }

            int sumTesting = 60;
            double overallAccuracy = (double)sumDiagonal / sumTesting;
            overallAccuracy *= 100;
            if (Math.Floor(overallAccuracy) == 0)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j <= num_of_clusters; j++)
                    {
                        temp_weights[i, j] = temp_weights[i, j];
                    }
                }
            }
            overallAccuracy = Math.Round(overallAccuracy, 2);
            return overallAccuracy.ToString() + "%";
        }

        public void classifySample(Matrix<double> sample)
        {
            for (int j = 1; j <= num_of_clusters; j++)
            {
                double r = euclidenDisance(sample, centroids[j]);
                phi[j] = Math.Exp((-r * r) / (2 * variances[j]));
            }

            double max = -1e9;
            int y = 0;
            for (int j = 1; j <= 3; j++)
            {
                double sum = 0;
                for (int k = 1; k <= num_of_clusters; k++)
                {
                    sum += phi[k] * weights[j, k];
                }
                if (sum > max)
                {
                    max = sum;
                    y = j;
                }
            }

            MessageBox.Show("The sample belongs to class " + y.ToString());
        }

        public void startTesting()
        {
            for (int class_num = 1; class_num <= 3; class_num++)
            {
                for (int i = 0; i < testing_set[class_num].Count; i++)
                {
                    for (int j = 1; j <= num_of_clusters; j++)
                    {
                        double r = euclidenDisance(testing_set[class_num][i], centroids[j]);
                        phi[j] = Math.Exp((-r * r) / (2 * variances[j]));
                    }

                    double max = -1e9;
                    int y = 0;
                    for (int j = 1; j <= 3; j++)
                    {
                        double sum = 0;
                        for (int k = 1; k <= num_of_clusters; k++)
                        {
                            sum += phi[k] * weights[j, k];
                        }
                        if (sum > max)
                        {
                            max = sum;
                            y = j;
                        }
                    }

                    int error = class_num - y;
                    confusion_matrix[class_num, y]++;
                }
            }
        }

        public void updateWeights(int error)
        {
            for (int j = 1; j <= num_of_clusters; j++)
            {
                for (int i = 1; i <= 3; i++)
                {
                    weights[i, j] += eta * error * phi[j];
                }
            }
        }

        public void initWeights()
        {
            Random rnd = new Random();

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j <= num_of_clusters; j++)
                {
                    weights[i, j] = rnd.Next(1, 100);
                }
            }

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j <= num_of_clusters; j++)
                {
                    temp_weights[i, j] = weights[i, j];
                }
            }
        }

        public void startTraining()
        {
            for (int e = 0; e < epochs; e++)
            {
                for (int class_num = 1; class_num <= 3; class_num++)
                {
                    for (int i = 0; i < training_set[class_num].Count; i++)
                    {
                        for (int j = 1; j <= num_of_clusters; j++)
                        {
                            double r = euclidenDisance(training_set[class_num][i], centroids[j]);
                            phi[j] = Math.Exp((-r * r) / (2 * variances[j]));
                        }

                        double max = -1e9;
                        int y = 0;
                        for (int j = 1; j <= 3; j++)
                        {
                            double sum = 0;
                            for (int k = 1; k <= num_of_clusters; k++)
                            {
                                sum += phi[k] * weights[j, k];
                            }
                            if (sum > max)
                            {
                                max = sum;
                                y = j;
                            }
                        }

                        int error = class_num - y;
                        updateWeights(error);
                    }
                }
            }
        }

        public void calculateVarainces()
        {
            for (int c = 1; c <= num_of_clusters; c++)
            {
                double sum = 0;

                for (int i = 0; i < clusters[c].Count; i++)
                    sum += euclidenDisance(clusters[c][i], centroids[c]);

                variances[c] = sum / clusters[c].Count;
            }
        }

        public void preprocessDataSet()
        {
            Matrix<double> mean = Matrix<double>.Build.Dense(4, 1);

            for (int class_num = 1; class_num <= 3; class_num++)
            {
                for (int i = 0; i < training_set[class_num].Count; i++)
                {
                    for (int j = 0; j < 4; j++)
                        mean[j, 0] += training_set[class_num][i][j, 0];
                }

                for (int i = 0; i < testing_set[class_num].Count; i++)
                {
                    for (int j = 0; j < 4; j++)
                        mean[j, 0] += testing_set[class_num][i][j, 0];
                }
            }

            for (int i = 0; i < 4; i++)
                mean[i, 0] /= 150;

            Matrix<double> max = Matrix<double>.Build.Dense(4, 1);

            for (int i = 0; i < 4; i++)
                max[i, 0] = -1e9;

            for (int class_num = 1; class_num <= 3; class_num++)
            {
                for (int i = 0; i < training_set[class_num].Count; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        training_set[class_num][i][j, 0] -= mean[j, 0];
                        max[j, 0] = Math.Max(max[j, 0], training_set[class_num][i][j, 0]);
                    }
                }

                for (int i = 0; i < testing_set[class_num].Count; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        testing_set[class_num][i][j, 0] -= mean[j, 0];
                        max[j, 0] = Math.Max(max[j, 0], testing_set[class_num][i][j, 0]);
                    }
                }
            }

            for (int class_num = 1; class_num <= 3; class_num++)
            {
                for (int i = 0; i < training_set[class_num].Count; i++)
                {
                    for (int j = 0; j < 4; j++)
                        training_set[class_num][i][j, 0] /= max[j, 0];
                }

                for (int i = 0; i < testing_set[class_num].Count; i++)
                {
                    for (int j = 0; j < 4; j++)
                        testing_set[class_num][i][j, 0] /= max[j, 0];
                }
            }
        }

        public double euclidenDisance(Matrix<double> sample, Matrix<double> centroid)
        {
            double dist = 0;
            for (int i = 0; i < 4; i++)
                dist += Math.Pow(sample[i, 0] - centroid[i, 0], 2);
            dist = Math.Sqrt(dist);
            return dist;
        }

        public void initCentroids()
        {
            Random rnd = new Random();
            bool[,] vis = new bool[4, 30];

            for (int i = 1; i <= num_of_clusters; i++)
            {
                int class_num, sample_num;
                while (true)
                {
                    class_num = rnd.Next(1, 4);
                    sample_num = rnd.Next(30);
                    if (!vis[class_num, sample_num])
                    {
                        vis[class_num, sample_num] = true;
                        break;
                    }
                }
                centroids[i] = training_set[class_num][sample_num];
            }
        }

        public void k_means()
        {
            int num_of_iterations = 0;
            while (num_of_iterations < 100)
            {
                for (int i = 1; i <= num_of_clusters; i++)
                    clusters[i].Clear();

                for (int class_num = 1; class_num <= 3; class_num++)
                {
                    for (int i = 0; i < training_set[class_num].Count; i++)
                    {
                        double min = 1e9;
                        int sample_cluster = 0;
                        for (int j = 1; j <= num_of_clusters; j++)
                        {
                            double dist = euclidenDisance(training_set[class_num][i], centroids[j]);
                            if (dist < min)
                            {
                                min = dist;
                                sample_cluster = j;
                            }
                        }

                        clusters[sample_cluster].Add(training_set[class_num][i]);
                    }
                }

                bool flag = false;

                for (int cluster_num = 1; cluster_num <= num_of_clusters; cluster_num++)
                {
                    Matrix<double> mean = Matrix<double>.Build.Dense(4, 1);

                    for (int i = 0; i < clusters[cluster_num].Count; i++)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            mean[j, 0] += clusters[cluster_num][i][j, 0];
                        }
                    }

                    for (int i = 0; i < 4; i++)
                        mean[i, 0] /= clusters[cluster_num].Count;

                    //If any centroid at the current iteration differs from the centroid of the previous one, then we continue k-means algorithm
                    //If the centroids of the current iteration are identical to the previous one, then we stop k-means algorithm
                    if (mean != centroids[cluster_num])
                        flag = true;

                    centroids[cluster_num] = mean;
                }
                if (!flag)
                    break;
                num_of_iterations++;
            }
        }

        public void readData()
        {
            int counter = 0, class_num = 1;
            string line;
            System.IO.StreamReader file =
                new System.IO.StreamReader(@"E:\College\Neural Network\Labs\Lab 2\activation function\Iris Data.txt");
            file.ReadLine();
            while ((line = file.ReadLine()) != null)
            {
                Console.WriteLine(line);
                counter++;
                string[] tmp = line.Split(',');
                Matrix<double> sample = Matrix<double>.Build.Dense(4, 1);
                int index = 0;
                for (int i = 0; i < 4; i++)
                    sample[index++, 0] = double.Parse(tmp[i]);
                if (counter > 30)
                    testing_set[class_num].Add(sample);
                else
                    training_set[class_num].Add(sample);
                if (counter == 50)
                {
                    class_num++;
                    counter = 0;
                }
            }
            file.Close();
        }
    }
}
