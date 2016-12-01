using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using System.Windows.Forms;

namespace Neural_Task_2
{
    class Regression
    {
        int c1, c2;
        int f1, f2;
        int error;
        Matrix<double> w;
        double lamda;
        Matrix<double> I;
        Matrix<double> mean0;
        List<Matrix<double>>[] training_set, testing_set;
        int[,] confusion_matrix;
        DataGridView dataGridView;

        public Regression(int c1, int c2, int f1, int f2, ref DataGridView dataGridView)
        {
            this.c1 = c1;
            this.c2 = c2;
            this.f1 = f1;
            this.f2 = f2;
            this.dataGridView = dataGridView;
            w = Matrix<double>.Build.Dense(2, 1);
            lamda = 0.1;
            I = Matrix<double>.Build.DenseIdentity(2, 2);
            mean0 = Matrix<double>.Build.Dense(2, 1);
            training_set = new List<Matrix<double>>[4];
            testing_set = new List<Matrix<double>>[4];
            for (int i = 0; i < 4; i++)
            {
                training_set[i] = new List<Matrix<double>>();
                testing_set[i] = new List<Matrix<double>>();
            }
            confusion_matrix = new int[5, 5];
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

        public double calculateOverallAccuracy()
        {
            int sumDiagonal = 0;
            for (int i = 1; i <= 3; i++)
            {
                sumDiagonal += confusion_matrix[i, i];
            }

            int sumTesting = 40;
            double overallAccuracy = (double)sumDiagonal / sumTesting;
            overallAccuracy *= 100;
            overallAccuracy = Math.Round(overallAccuracy, 2);
            //textBox.Text = overallAccuracy.ToString() + "%";
            //textBox2.Text = error.ToString();
            return overallAccuracy;
        }

        public void normalizeTesting()
        {
            mean0 = Matrix<double>.Build.Dense(2, 1);
            foreach (Matrix<double> sample in testing_set[c1])
            {
                for (int i = 0; i < 2; i++)
                    mean0[i, 0] += sample[i, 0];
            }
            foreach (Matrix<double> sample in testing_set[c2])
            {
                for (int i = 0; i < 2; i++)
                    mean0[i, 0] += sample[i, 0];
            }

            for (int i = 0; i < 2; i++)
                mean0[i, 0] /= 40;

            Matrix<double> max = Matrix<double>.Build.Dense(2, 1);

            //foreach (Matrix<double> sample in testing_set[c1])
            //{
            //    for (int i = 0; i < 2; i++)
            //    {
            //        sample[i, 0] -= mean0[i, 0];
            //        sample[i, 0] = Math.Abs(sample[i, 0]);
            //        max[i, 0] = Math.Max(max[i, 0], sample[i, 0]);
            //    }
            //}
            for (int i = 0; i < testing_set[c1].Count; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    testing_set[c1][i][j, 0] -= mean0[j, 0];
                    testing_set[c1][i][j, 0] -= Math.Abs(testing_set[c1][i][j, 0]);
                    max[j, 0] = Math.Max(max[j, 0], testing_set[c1][i][j, 0]);
                }
            }
            for (int i = 0; i < testing_set[c2].Count; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    testing_set[c2][i][j, 0] -= mean0[j, 0];
                    testing_set[c2][i][j, 0] -= Math.Abs(testing_set[c2][i][j, 0]);
                    max[j, 0] = Math.Max(max[j, 0], testing_set[c2][i][j, 0]);
                }
            }

            for (int i = 0; i < testing_set[c1].Count; i++)
            {
                for (int j = 0; j < 2; j++)
                    testing_set[c1][i][j, 0] /= max[j, 0];
            }
            for (int i = 0; i < testing_set[c2].Count; i++)
            {
                for (int j = 0; j < 2; j++)
                    testing_set[c2][i][j, 0] /= max[j, 0];
            }
        }

        public void normalizeTraining()
        {
            foreach (Matrix<double> sample in training_set[c1])
            {
                for (int i = 0; i < 2; i++)
                    mean0[i, 0] += sample[i, 0];
            }
            foreach (Matrix<double> sample in training_set[c2])
            {
                for (int i = 0; i < 2; i++)
                    mean0[i, 0] += sample[i, 0];
            }

            for (int i = 0; i < 2; i++)
                mean0[i, 0] /= 60;

            Matrix<double> max = Matrix<double>.Build.Dense(2, 1);

            //foreach (Matrix<double> sample in testing_set[c1])
            //{
            //    for (int i = 0; i < 2; i++)
            //    {
            //        sample[i, 0] -= mean0[i, 0];
            //        sample[i, 0] = Math.Abs(sample[i, 0]);
            //        max[i, 0] = Math.Max(max[i, 0], sample[i, 0]);
            //    }
            //}
            for (int i = 0; i < training_set[c1].Count; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    training_set[c1][i][j, 0] -= mean0[j, 0];
                    training_set[c1][i][j, 0] -= Math.Abs(training_set[c1][i][j, 0]);
                    max[j, 0] = Math.Max(max[j, 0], training_set[c1][i][j, 0]);
                }
            }
            for (int i = 0; i < training_set[c2].Count; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    training_set[c2][i][j, 0] -= mean0[j, 0];
                    training_set[c2][i][j, 0] -= Math.Abs(training_set[c2][i][j, 0]);
                    max[j, 0] = Math.Max(max[j, 0], training_set[c2][i][j, 0]);
                }
            }

            for (int i = 0; i < training_set[c1].Count; i++)
            {
                for (int j = 0; j < 2; j++)
                    training_set[c1][i][j, 0] /= max[j, 0];
            }
            for (int i = 0; i < training_set[c2].Count; i++)
            {
                for (int j = 0; j < 2; j++)
                    training_set[c2][i][j, 0] /= max[j, 0];
            }
        }

        public void linearRegression()
        {
            Matrix<double> x = Matrix<double>.Build.Dense(2, 60);
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < training_set[c1].Count; j++)
                    x[i, j] = training_set[c1][j][i, 0];
            }
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < training_set[c2].Count; j++)
                    x[i, j + 30] = training_set[c2][j][i, 0];
            }
            x = x.Transpose();
            Matrix<double> RXX = x.Transpose().Multiply(x);
            RXX = RXX.Negate();
            Matrix<double> LI = I.Multiply(lamda);
            Matrix<double> Inv = RXX.Add(LI);
            Inv = Inv.Inverse();
            Matrix<double> d = Matrix<double>.Build.Dense(1, 60);
            for (int i = 0; i < 60; i++)
            {
                if (i < 30)
                    d[0, i] = 1;
                else
                    d[0, i] = -1;
            }
            d = d.Transpose();
            Matrix<double> rdx = x.Transpose().Multiply(d);
            rdx = rdx.Negate();
            w = Inv.Multiply(rdx);
        }
        public int activationFunction(double vk)
        {
            if (vk > 0)
                return 1;
            else if (vk == 0)
                return 0;
            else
                return -1;
        }

        public double adder(Matrix<double> x)
        {
            Matrix<double> w_transpose = w.Transpose();
            w_transpose = w_transpose.Multiply(x);
            double vk = w_transpose[0, 0];
            return vk;
        }

        public void startTesting()
        {
            for (int i = 0; i < testing_set[c1].Count; i++)
            {
                Matrix<double> x = testing_set[c1][i];
                double vk = adder(x);
                int y = activationFunction(vk);
                if (y != c1)
                    error++;
                if (y == 1)
                    confusion_matrix[c1, c1]++;
                else
                    confusion_matrix[c1, c2]++;
            }
            for (int i = 0; i < testing_set[c2].Count; i++)
            {
                Matrix<double> x = testing_set[c2][i];
                double vk = adder(x);
                int y = activationFunction(vk);
                if (y != c2)
                    error++;
                if (y == -1)
                    confusion_matrix[c2, c2]++;
                else
                    confusion_matrix[c2, c1]++;
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
                Matrix<double> sample = Matrix<double>.Build.Dense(2, 1);
                int index = 0;
                for (int i = 1; i <= 4; i++)
                {
                    if (i == f1 || i == f2)
                        sample[index++, 0] = double.Parse(tmp[i - 1]);
                }
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
