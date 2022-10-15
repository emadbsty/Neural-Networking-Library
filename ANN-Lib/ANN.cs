using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace ANN_Lib
{
    // This Neural Network Library Devloped by Emad Bsty Syria +963988553257
    // 


    public class ANN
    {
        //ANN(InputsX,Hidden Neurals,Output Neurals)
        public double[] InputsX;
        public List<Neural[]> HN;
        public Neural[] ON;
        private double learnRate;
        public ActivationFunction haf;
        public ActivationFunction oaf;
        public double learnrate
        {
            get { return learnRate; }
            set { learnRate = value; }
        }
        public enum ActivationFunction
        {
            Sigmoid,
            ReLU,
            TanH,
            SoftMax,
        }
        public ANN(int x, int[] h, ActivationFunction haf, int o, ActivationFunction oaf)
        {
            learnRate = 0.6d;
            InputsX = new double[x];
            HN = new List<Neural[]>();
            Neural[] hn;
            ON = new Neural[o];

            this.haf = haf;
            this.oaf = oaf;

            for (int i = 0; i < h.Length; i++)
            {
                hn = new Neural[h[i]];
                HN.Add(hn);
            }
            for (int i = 1; i < HN.Count; i++)
            {
                for (int j = 0; j < HN[i].Length; j++)
                {
                    HN[i][j] = new Neural(HN[i - 1].Length);
                }
            }

            for (int i = 0; i < HN[0].Length; i++)
            {
                HN[0][i] = new Neural(x);
            }
            for (int i = 0; i < o; i++)
            {
                ON[i] = new Neural(HN[HN.Count - 1].Length);
            }
        }
        public void Update(double[] Target)
        {
            for (int i = 0; i < ON.Length; i++)
            {
                if (oaf == ActivationFunction.ReLU)
                {
                    //ReLU
                    ON[i].Error = DerivatReLU(ON[i].y) * (Target[i] - ON[i].y);
                }
                else if (oaf == ActivationFunction.Sigmoid)
                {
                    // Segmoid
                    ON[i].Error = ON[i].y * (1.0f - ON[i].y) * (Target[i] - ON[i].y);
                }
                else if (oaf == ActivationFunction.TanH)
                {
                    // TanH
                    ON[i].Error = DerivatTanh(ON[i].y) * (Target[i] - ON[i].y); ;
                }
            }
            for (int i = 0; i < HN[HN.Count - 1].Length; i++)
            {
                HN[HN.Count - 1][i].Error = 0.0f;
                for (int j = 0; j < ON.Length; j++)
                {
                    HN[HN.Count - 1][i].Error += ON[j].w[i] * ON[j].Error;
                }
                if (haf == ActivationFunction.ReLU)
                {
                    //ReLU
                    HN[HN.Count - 1][i].Error = HN[HN.Count - 1][i].Error * DerivatReLU(HN[HN.Count - 1][i].y);
                }
                else if (haf == ActivationFunction.Sigmoid)
                {
                    // Segmoid
                    HN[HN.Count - 1][i].Error = HN[HN.Count - 1][i].Error * HN[HN.Count - 1][i].y * (1.0f - HN[HN.Count - 1][i].y);
                }
                else if (haf == ActivationFunction.TanH)
                {
                    // TanH
                    HN[HN.Count - 1][i].Error = HN[HN.Count - 1][i].Error * DerivatTanh(HN[HN.Count - 1][i].y);
                }
            }
            for (int k = HN.Count - 2; k >= 0; k--)
            {
                for (int i = 0; i < HN[k].Length; i++)
                {
                    HN[k][i].Error = 0.0f;
                    for (int j = 0; j < HN[k + 1].Length; j++)
                    {
                        HN[k][i].Error += HN[k + 1][j].w[i] * HN[k + 1][j].Error;
                    }
                    if (haf == ActivationFunction.ReLU)
                    {
                        //ReLU
                        HN[k][i].Error = HN[k][i].Error * DerivatReLU(HN[k][i].y);
                    }
                    else if (haf == ActivationFunction.Sigmoid)
                    {
                        // Segmoid
                        HN[k][i].Error = HN[k][i].Error * HN[k][i].y * (1.0f - HN[k][i].y);
                    }
                    else if (haf == ActivationFunction.TanH)
                    {
                        // TanH
                        HN[k][i].Error = HN[k][i].Error * DerivatTanh(HN[k][i].y);
                    }
                }
            }
            for (int j = 0; j < ON.Length; j++)
            {
                for (int i = 0; i < ON[j].w.Length; i++)
                {
                    ON[j].w[i] = ON[j].w[i] + learnRate * HN[HN.Count - 1][i].y * ON[j].Error;
                }
                ON[j].bias += learnRate * ON[j].Error;
            }
            for (int k = 1; k < HN.Count; k++)
            {
                for (int i = 0; i < HN[k].Length; i++)
                {
                    for (int j = 0; j < HN[k][i].w.Length; j++)
                    {
                        HN[k][i].w[j] = HN[k][i].w[j] + learnRate * HN[k - 1][j].y * HN[k][i].Error;
                    }
                    HN[k][i].bias += learnRate * HN[k][i].Error;
                }
            }
            for (int i = 0; i < HN[0].Length; i++)
            {
                for (int j = 0; j < HN[0][i].w.Length; j++)
                {
                    HN[0][i].w[j] = HN[0][i].w[j] + learnRate * InputsX[j] * HN[0][i].Error;
                }
                HN[0][i].bias += learnRate * HN[0][i].Error;
            }

            //********************** update other hidden 
        }
        public void Process(double[] inputsx)
        {
            InputsX = inputsx;
            double aux;
            //*******************************************
            for (int i = 0; i < HN[0].Length; i++)
            {
                aux = 0.0d;
                for (int j = 0; j < HN[0][i].w.Length; j++)
                {
                    aux += InputsX[j] * HN[0][i].w[j];
                }
                aux += HN[0][i].bias;
                HN[0][i].net = aux;
                if (haf == ActivationFunction.ReLU)
                {
                    //ReLU
                    HN[0][i].y = ReLU(aux);
                }
                else if (haf == ActivationFunction.Sigmoid)
                {
                    // Segmoid
                    HN[0][i].y = sigmoid(aux);
                }
                else if (haf == ActivationFunction.TanH)
                {
                    // TanH
                    HN[0][i].y = Tanh(aux);
                }
            }
            //*******************************************
            for (int k = 1; k < HN.Count; k++)
            {
                for (int i = 0; i < HN[k].Length; i++)
                {
                    aux = 0.0d;
                    for (int j = 0; j < HN[k][i].w.Length; j++)
                    {
                        aux += HN[k - 1][j].y * HN[k][i].w[j];
                    }
                    aux += HN[k][i].bias;
                    HN[k][i].net = aux;
                    if (haf == ActivationFunction.ReLU)
                    {
                        //ReLU
                        HN[k][i].y = ReLU(aux);
                    }
                    else if (haf == ActivationFunction.Sigmoid)
                    {
                        // Segmoid
                        HN[k][i].y = sigmoid(aux);
                    }
                    else if (haf == ActivationFunction.TanH)
                    {
                        // TanH
                        HN[k][i].y = Tanh(aux);
                    }

                }
            }

            //*******************************************
            double[] yy = new double[ON.Length];
            for (int j = 0; j < ON.Length; j++)
            {
                aux = 0.0f;
                for (int i = 0; i < ON[j].w.Length; i++)
                {
                    aux += HN[HN.Count - 1][i].y * ON[j].w[i];
                }
                aux += ON[j].bias;
                ON[j].net = aux;
                if (oaf == ActivationFunction.ReLU)
                {
                    //ReLU
                    ON[j].y = ReLU(aux);
                }
                else if (oaf == ActivationFunction.ReLU)
                {
                    // Segmoid
                    ON[j].y = sigmoid(aux);
                }
                else if (haf == ActivationFunction.TanH)
                {
                    // TanH
                    ON[j].y = Tanh(aux);
                }
                yy[j] = ON[j].y;
            }
            //*******************************************
            //return yy;
        }
        public double sigmoid(double xx)
        {
            return (1.0d / (1.0d + (double)Math.Exp(-xx)));
        }
        public double Derivatsigmoid(double xx)
        {
            return (sigmoid(xx) * (1.0d - sigmoid(xx)));
        }
        double aux = 0.0d;
        public double ReLU(double xx)
        {
            //if (xx > 0.0) return xx;
            //else return (aux * xx);
            return xx;
        }
        public double DerivatReLU(double xx)
        {
            //if (xx > 0.0) return 1.0d;
            //else return aux;
            return 1.0d;
        }
        public double Tanh(double x)
        {
            return ((Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x)));
        }
        public double DerivatTanh(double x)
        {
            return (1.0d - (Tanh(x) * Tanh(x)));
        }
        public double[] SoftMax(double[] arr)
        {
            double sum = 0.0d;
            double[] aux = new double[arr.Length];
            for (int i = 0; i < arr.Length; i++)
            {
                aux[i] = Math.Pow(Math.E, arr[i]);
                sum += aux[i];
            }
            for (int i = 0; i < arr.Length; i++)
            {
                aux[i] = aux[i] / sum;
            }
            return aux;
        }
        public double DerivatSoftMax(int index, double[] arr)
        {
            double sum = 0.0d;
            double[] aux = new double[arr.Length];
            double po = 0.0d;
            for (int i = 0; i < arr.Length; i++)
            {
                sum += Math.Pow(Math.E, arr[i]);
            }
            sum = sum * sum;
            for (int i = 0; (i < arr.Length) && (i != index); i++)
            {
                po += Math.Pow(Math.E, arr[i]);
            }
            po = (arr[index] * po) / sum;
            return po;
        }
        public double[] LogLoss(double[] y, double[] o)
        {
            double[] ll = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
            {
                ll[i] = -1.0d * y[i] * Math.Log10(o[i]) - (1.0d - y[i]) * Math.Log10(1.0d - o[i]);
            }
            return ll;
        }
        public double[] DerivatLogLoss(double[] y, double[] o)
        {
            double[] ll = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
            {
                ll[i] = -1.0d * y[i] * (1.0d / o[i]) - (1.0d - y[i]) * (1.0d / (1.0d - o[i]));
            }
            return ll;
        }
        public class Neural
        {
            public double[] w;
            public double bias;
            public double Error;
            public double net;
            public double y;
            static Random rand = new Random();

            public Neural(int inputs)
            {
                w = new double[inputs];
                for (int i = 0; i < inputs; i++)
                {
                    w[i] = ((double)rand.Next(-3000, 3000) / 5000.0f);
                }
                bias = ((double)rand.Next(-2000, 2000) / 5000.0f);
            }
        }
    }


    public class SOM_Network
    {
        public Point dimension;
        public float learningRate;
        SOM_Nodes[,] nodes;

        public SOM_Nodes[,] Nodes
        {
            get { return nodes; }
        }
        public SOM_Network(Point dim, int inputSize)
        {
            this.dimension = dim;
            nodes = new SOM_Nodes[dim.X, dim.Y];
            for (int i = 0; i < dimension.Y; i++)
            {
                for (int j = 0; j < dimension.X; j++)
                {
                    nodes[j, i] = new SOM_Nodes(new Point(j, i), inputSize);
                }
            }
            learningRate = 0.5f;
        }
        public void CalcDistance(Color inputs)
        {
            for (int i = 0; i < dimension.Y; i++)
            {
                for (int j = 0; j < dimension.X; j++)
                {
                    nodes[j, i].Distance = nodes[j, i].Distance_Clac(inputs);
                    nodes[j, i].color = Color.FromArgb((int)nodes[j, i].Weights[0]
                        , (int)nodes[j, i].Weights[1]
                        , (int)nodes[j, i].Weights[2]);
                }
            }
        }
        public SOM_Nodes Get_BMU(Color inputs)
        {
            SOM_Nodes node = nodes[0, 0];
            float min_dis = nodes[0, 0].Distance;
            int im = 0, jm = 0;
            for (int i = 0; i < dimension.Y; i++)
            {
                for (int j = 0; j < dimension.X; j++)
                {
                    if (nodes[j, i].Distance < min_dis)
                    {
                        min_dis = nodes[j, i].Distance;
                        node = nodes[j, i];
                        jm = j; im = i;
                    }
                }
            }
            nodes[jm, im].color = inputs;
            nodes[jm, im].Distance = 0.0f;
            node = nodes[jm, im];
            return node;
        }
        public void Update(List<Point> neighboring, Color inputs, float LR)
        {
            for (int i = 0; i < neighboring.Count; i++)
            {
                nodes[neighboring[i].X, neighboring[i].Y].Weights[0] = nodes[neighboring[i].X, neighboring[i].Y].Weights[0] + LR * (inputs.R - nodes[neighboring[i].X, neighboring[i].Y].Weights[0]);
                nodes[neighboring[i].X, neighboring[i].Y].Weights[1] = nodes[neighboring[i].X, neighboring[i].Y].Weights[1] + LR * (inputs.G - nodes[neighboring[i].X, neighboring[i].Y].Weights[1]);
                nodes[neighboring[i].X, neighboring[i].Y].Weights[2] = nodes[neighboring[i].X, neighboring[i].Y].Weights[2] + LR * (inputs.B - nodes[neighboring[i].X, neighboring[i].Y].Weights[2]);
                nodes[neighboring[i].X, neighboring[i].Y].color = Color.FromArgb((int)nodes[neighboring[i].X, neighboring[i].Y].Weights[0], (int)nodes[neighboring[i].X, neighboring[i].Y].Weights[1], (int)nodes[neighboring[i].X, neighboring[i].Y].Weights[2]);
            }
        }
        public void UpdateOne(Point neighboring, Color inputs, float lr)
        {
            nodes[neighboring.X, neighboring.Y].Weights[0] = nodes[neighboring.X, neighboring.Y].Weights[0] + lr * (inputs.R - nodes[neighboring.X, neighboring.Y].Weights[0]);
            nodes[neighboring.X, neighboring.Y].Weights[1] = nodes[neighboring.X, neighboring.Y].Weights[1] + lr * (inputs.G - nodes[neighboring.X, neighboring.Y].Weights[1]);
            nodes[neighboring.X, neighboring.Y].Weights[2] = nodes[neighboring.X, neighboring.Y].Weights[2] + lr * (inputs.B - nodes[neighboring.X, neighboring.Y].Weights[2]);
            nodes[neighboring.X, neighboring.Y].color = Color.FromArgb((int)nodes[neighboring.X, neighboring.Y].Weights[0], (int)nodes[neighboring.X, neighboring.Y].Weights[1], (int)nodes[neighboring.X, neighboring.Y].Weights[2]);

        }
        public List<Point> Get_neighboring(SOM_Nodes BMU, int radius)
        {
            List<Point> pp = new List<Point>();
            Point aux;
            for (int i = -radius; i <= radius; i++)
            {
                for (int j = -radius; j <= radius; j++)
                {
                    int XX = BMU.Location.X + j;
                    int YY = BMU.Location.Y + i;

                    if (XX >= this.dimension.X)
                        break;
                    if (XX < 0)
                        break;
                    if (YY >= this.dimension.Y)
                        break;
                    if (YY < 0)
                        break;

                    aux = new Point(XX, YY);
                    if (!FindInList(pp, aux))
                        pp.Add(aux);
                }
            }
            return pp;
        }
        public void Get_neighboring_Update(SOM_Nodes BMU, Color inputs, int radius, double factor)
        {
            Point aux;
            for (int i = -radius; i <= radius; i++)
            {
                for (int j = -radius; j <= radius; j++)
                {
                    int XX = BMU.Location.X + j;
                    int YY = BMU.Location.Y + i;

                    if (XX >= this.dimension.X)
                        break;
                    if (XX < 0)
                        break;
                    if (YY >= this.dimension.Y)
                        break;
                    if (YY < 0)
                        break;
                    double ra = Math.Sqrt((i * i) + (j * j));
                    float lr = learningRate * (float)Math.Exp((float)(-ra) / (float)(factor * radius));
                    aux = new Point(XX, YY); UpdateOne(aux, inputs, lr);
                }
            }
        }
        public bool FindInList(List<Point> pp, Point p)
        {
            bool b = false;
            for (int i = 0; i < pp.Count; i++)
            {
                if ((pp[i].X == p.X) & (pp[i].Y == p.Y))
                    b = true;
            }
            return b;
        }
        public class SOM_Nodes
        {
            private float[] w;
            private Point loc;
            private float distance;
            private Color c;
            static Random rand = new Random();

            public Point Location
            {
                get { return loc; }
                set { loc = value; }
            }
            public Color color
            {
                get { return c; }
                set { c = value; }
            }
            public float Distance
            {
                get { return distance; }
                set { distance = value; }
            }
            public float[] Weights
            {
                get { return w; }
                set { w = value; }
            }
            public SOM_Nodes(Point loc, int inputSize)
            {
                this.w = new float[inputSize];
                this.loc = loc;
                for (int i = 0; i < w.Length; i++)
                {
                    w[i] = (float)(rand.NextDouble() * 255.0f);
                }
            }
            public float Distance_Clac(Color inputs)
            {
                float dis = 0.0f;
                dis += (inputs.R - this.w[0]) * (inputs.R - this.w[0]);
                dis += (inputs.G - this.w[1]) * (inputs.G - this.w[1]);
                dis += (inputs.B - this.w[2]) * (inputs.B - this.w[2]);
                dis = dis / 3.0f;
                return (float)Math.Sqrt(dis);
            }
        }
    }
    public class CNN
    {
        public float[,] Conv(float[,] data, float[,] filter)
        {
            int w = data.GetLength(0) - (filter.GetLength(0) - 1);
            int h = data.GetLength(1) - (filter.GetLength(1) - 1);
            float[,] aux = new float[w, h];
            for (int i = 0; i < aux.GetLength(1); i++)
            {
                for (int j = 0; j < aux.GetLength(0); j++)
                {
                    int xf = j * filter.GetLength(0); int yf = i * filter.GetLength(1);
                    float sum = 0.0f;
                    for (int ii = i; ii < i + filter.GetLength(1); ii++)
                    {
                        for (int jj = j; jj < j + filter.GetLength(0); jj++)
                        {
                            sum += data[jj, ii] * filter[jj - j, ii - i];
                        }
                    }
                    sum = sum / (float)(filter.GetLength(0) * filter.GetLength(1));
                    aux[j, i] = sum;
                }
            }
            return aux;
        }
        public float[,] ReLu(float[,] data)
        {
            for (int i = 0; i < data.GetLength(1); i++)
            {
                for (int j = 0; j < data.GetLength(0); j++)
                {
                    if (data[j, i] < 0.0f)
                        data[j, i] = 0.0f;                    
                }
            }
            return data;
        }
        public float[,] Pooling(float[,] image, int w, int h)
        {
            float[,] aux = new float[image.GetLength(0) / w, image.GetLength(1) / h];
            float max = -1.0f; int ii = 0; int jj = 0;
            for (int i = 0; i < image.GetLength(0); i += h)
            {
                for (int j = 0; j < image.GetLength(1); j += w)
                {
                    for (int iii = 0; iii < h; iii++)
                    {
                        for (int jjj = 0; jjj < h; jjj++)
                        {
                            if (max < image[i + iii, j + jjj])
                                max = image[i + iii, j + jjj];
                        }
                    }
                    aux[ii, jj] = max;
                    jj++; max = -1.0f;
                }
                ii++; jj = 0; max = -1.0f;
            }
            return aux;
        }
        public float[,] AddPadding(float[,] image, int Pad)
        {
            
            float[,] aux = new float[image.GetLength(0) + 2 * Pad, image.GetLength(1) + 2 * Pad];
            for (int i = 0; i < aux.GetLength(0); i++)
            {
                for (int j = 0; j < aux.GetLength(1); j++)
                {
                    aux[j, i] = -1.0f;
                }
            }
            for (int i = Pad; i < aux.GetLength(1) - Pad; i++)
            {
                for (int j = Pad; j < aux.GetLength(0) - Pad; j++)
                {
                    aux[j, i] = image[j - Pad, i - Pad];
                }
            }
            return aux;
        }
        public float[,] RemovePadding(float[,] image, int Pad)
        {
            float[,] aux = new float[image.GetLength(0) - 2 * Pad, image.GetLength(1) - 2 * Pad];

            for (int i = 0; i < aux.GetLength(1); i++)
            {
                for (int j = 0; j < aux.GetLength(0); j++)
                {
                    aux[j, i] = image[j + Pad, i + Pad];
                }
            }

            return aux;
        }
    }
}
