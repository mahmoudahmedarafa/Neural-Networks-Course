using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Neural_Task_1
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            double eta = double.Parse(textBox1.Text);
            int epochs = int.Parse(textBox2.Text);
            int c1 = int.Parse(textBox3.Text);
            int c2 = int.Parse(textBox4.Text);
            int f1 = int.Parse(textBox7.Text);
            int f2 = int.Parse(textBox8.Text);
            Perceptron perceptron = new Perceptron(ref dataGridView1, ref textBox6, ref textBox5,
                eta, epochs, c1, c2, f1, f2);
            perceptron.readData();
            perceptron.startTraining();
            perceptron.startTesting();
            perceptron.prepareConfusionMatrix();
            perceptron.displayConfusionMatrix();
            perceptron.calculateOverallAccuracy();
        }

        private void label4_Click(object sender, EventArgs e)
        {

        }

        private void textBox8_TextChanged(object sender, EventArgs e)
        {

        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }
    }
}
