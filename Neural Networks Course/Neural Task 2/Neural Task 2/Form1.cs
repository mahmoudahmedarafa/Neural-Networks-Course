using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Neural_Task_2
{
    public partial class Form1 : Form
    {
        int _selectedIndex;

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            comboBox1.Items.Add("Batch Perceptron");
            comboBox1.Items.Add("Least Mean Square");
            comboBox1.Items.Add("Linear Regression");
        }

        private void button1_Click(object sender, EventArgs e)
        {
            int epochs = int.Parse(textBox2.Text);
            int c1 = int.Parse(textBox3.Text);
            int c2 = int.Parse(textBox4.Text);
            int f1 = int.Parse(textBox7.Text);
            int f2 = int.Parse(textBox8.Text);
            if (_selectedIndex == 0)
            {
                //Batch Perceptron
                double eta = 0;
                double best_accuracy = 0;
                int[,] confusion_matrix = new int[5, 5];
                for (int i = 5; i <= 100; i += 5)
                {
                    //Perceptron perceptron = new Perceptron(ref dataGridView1, ref textBox6, ref textBox5,
                    //    i / 100.0, epochs, c1, c2, f1, f2);
                    Perceptron perceptron = new Perceptron(i / 100.0, epochs, c1, c2, f1, f2);
                    perceptron.readData();
                    perceptron.startTraining();
                    perceptron.startTesting();
                    int[,] tmp = perceptron.prepareConfusionMatrix();
                    //perceptron.displayConfusionMatrix();
                    double accuracy = perceptron.calculateOverallAccuracy();
                    if (accuracy > best_accuracy)
                    {
                        best_accuracy = accuracy;
                        eta = i / 100.0;
                        for (int j = 0; j < 5; j++)
                            for (int k = 0; k < 5; k++)
                                confusion_matrix[j, k] = tmp[j, k];
                    }
                }
                Perceptron.displayConfusionMatrix(ref dataGridView1, confusion_matrix);
                textBox1.Text = eta.ToString();
                textBox6.Text = best_accuracy.ToString() + "%";
            }
            else if (_selectedIndex == 1)
            {
                //Least Mean Square
            }
            else
            {
                //Linear Regression
                Regression regression = new Regression(c1, c2, f1, f2, ref dataGridView1);
                regression.readData();
                regression.normalizeTraining();
                regression.normalizeTesting();
                regression.linearRegression();
                regression.startTesting();
                regression.prepareConfusionMatrix();
                regression.displayConfusionMatrix();
                double accuracy = regression.calculateOverallAccuracy();
                textBox6.Text = accuracy.ToString() + "%";
            }
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            _selectedIndex = comboBox1.SelectedIndex;
        }
    }
}
