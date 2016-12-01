using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using System.Windows.Forms;

namespace Neural_Task_1
{
    class Perceptron
    {
        Matrix<double> weight;
        int c1, c2;
        int f1, f2;
        //Arrays are 1-based
        List <Matrix<double>>[] training_set, testing_set;
        int error, epochs;
        double eta;
        int[,] confusion_matrix;
        DataGridView dataGridView;
        TextBox textBox, textBox2;

        public Perceptron(ref DataGridView dataGridView, ref TextBox textBox, ref TextBox textBox2, double eta, int epochs, int c1, int c2,
            int f1, int f2)
        {
            weight = Matrix<double>.Build.Dense(3, 1);
            training_set = new List<Matrix<double>>[4];
            testing_set = new List<Matrix<double>>[4];
            for (int i = 0; i < 4; i++)
            {
                training_set[i] = new List<Matrix<double>>();
                testing_set[i] = new List<Matrix<double>>();
            }
            confusion_matrix = new int[5, 5];
            this.dataGridView = dataGridView;
            this.textBox = textBox;
            this.textBox2 = textBox2;
            this.eta = eta;
            this.epochs = epochs;
            this.c1 = c1;
            this.c2 = c2;
            this.f1 = f1;
            this.f2 = f2;
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

        public void calculateOverallAccuracy()
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
            textBox.Text = overallAccuracy.ToString() + "%";
            textBox2.Text = error.ToString();
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
            Matrix<double> w_transpose = weight.Transpose();
            w_transpose = w_transpose.Multiply(x);
            double vk = w_transpose[0, 0];
            return vk;
        }

        public void startTraining()
        {
            for (int i = 0; i < epochs; i++)
            {
                for (int j = 0; j < training_set[c1].Count; j++)
                {
                    int d = 1;  //d = 1 if training sample belongs to C1 and d = -1 if it belongs to C2 
                    Matrix<double> x = training_set[c1][j];
                    double vk = adder(x);
                    int y = activationFunction(vk);
                    Matrix<double> w_new = weight.Add(x.Multiply(eta * (d - y)));
                    weight = w_new;
                }
                for (int j = 0; j < training_set[c2].Count; j++)
                {
                    int d = -1;  //d = -1 because it belongs to C2 
                    Matrix<double> x = training_set[c2][j];
                    double vk = adder(x);
                    int y = activationFunction(vk);
                    Matrix<double> w_new = weight.Add(x.Multiply(eta * (d - y)));
                    weight = w_new;
                }
            }
        }

        public void startTesting()
        {
            for (int j = 0; j < testing_set[c1].Count; j++)
            {
                int d = 1;
                Matrix<double> x = testing_set[c1][j];
                double vk = adder(x);
                int y = activationFunction(vk);
                if (y != d)
                {
                    confusion_matrix[c1, c2]++;
                    error++;
                }
                else
                    confusion_matrix[c1, c1]++;
            }
            for (int j = 0; j < testing_set[c2].Count; j++)
            {
                int d = -1;
                Matrix<double> x = testing_set[c2][j];
                double vk = adder(x);
                int y = activationFunction(vk);
                if (y != d)
                {
                    confusion_matrix[c2, c1]++;
                    error++;
                }
                else
                    confusion_matrix[c2, c2]++;
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
                Matrix<double> sample = Matrix<double>.Build.Dense(3, 1);
                sample[0, 0] = 1;
                int index = 1;
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
