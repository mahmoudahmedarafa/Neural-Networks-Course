using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Neural_Task_4
{
    public partial class Form1 : Form
    {
        RBF rbf;

        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            double eta = double.Parse(textBox1.Text);
            int epochs = int.Parse(textBox2.Text);
            int num_of_clusters = int.Parse(textBox3.Text);

            rbf = new RBF(eta, epochs, num_of_clusters, ref dataGridView1);

            rbf.readData();
            rbf.preprocessDataSet();
            rbf.initCentroids();
            rbf.initWeights();
            
            rbf.k_means();
            rbf.calculateVarainces();
            
            //rbf.calculateVarainces();
            rbf.startTraining();
            rbf.startTesting();
            rbf.prepareConfusionMatrix();
            rbf.displayConfusionMatrix();
            textBox4.Text = rbf.calculateOverallAccuracy();

            MessageBox.Show("Done!");
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            textBox1.Text = "0.9";
            textBox2.Text = "50";
            textBox3.Text = "2";
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Matrix<double> sample = Matrix<double>.Build.Dense(4, 1);

            sample[0, 0] = double.Parse(textBox5.Text);
            sample[1, 0] = double.Parse(textBox6.Text);
            sample[2, 0] = double.Parse(textBox7.Text);
            sample[3, 0] = double.Parse(textBox8.Text);

            rbf.classifySample(sample);
        }
    }
}
