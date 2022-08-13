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
        public float[] InputsX;
        public Neural[] HN;
        public Neural[] ON;
        private float learnRate;
        public float learnrate
        {
            get { return learnRate; }
            set { learnRate = value; }
        }
        public ANN(int x, int h, int o)
        {
            learnRate = 0.5f;
            InputsX = new float[x];
            HN = new Neural[h];
            ON = new Neural[o];
            for (int i = 0; i < h; i++)
            {
                HN[i] = new Neural(x);
            }
            for (int i = 0; i < o; i++)
            {
                ON[i] = new Neural(h);
            }
        }
        public void Update(float[] Target)
        {
            for (int i = 0; i < ON.Length; i++)
            {
                ON[i].Error = ON[i].y * (1.0f - ON[i].y) * (Target[i] - ON[i].y);
                //ON[i].Error = ON[i].net * (1.0f - ON[i].net) * (Target[i] - ON[i].y);
            }

            for (int i = 0; i < HN.Length; i++)
            {
                HN[i].Error = 0.0f;
                for (int j = 0; j < ON.Length; j++)
                {
                    HN[i].Error += ON[j].w[i] * ON[j].Error;
                }
                HN[i].Error = HN[i].Error * HN[i].y * (1.0f - HN[i].y);
                //HN[i].Error = HN[i].Error * HN[i].net * (1.0f - HN[i].net);
            }
            for (int j = 0; j < ON.Length; j++)
            {
                for (int i = 0; i < ON[j].w.Length; i++)
                {
                    ON[j].w[i] = ON[j].w[i] + HN[i].y * ON[j].Error;
                }
                ON[j].bias += ON[j].Error;
            }

            for (int i = 0; i < HN.Length; i++)
            {
                for (int j = 0; j < HN[i].w.Length; j++)
                {
                    HN[i].w[j] = HN[i].w[j] + InputsX[j] * HN[i].Error;
                }
                HN[i].bias += HN[i].Error;
            }
        }
        public float[] Process(float[] inputsx)
        {
            InputsX = inputsx;
            float aux;
            for (int i = 0; i < HN.Length; i++)
            {
                aux = 0.0f;
                for (int j = 0; j < HN[i].w.Length; j++)
                {
                    aux += InputsX[j] * HN[i].w[j];
                }
                aux += HN[i].bias;
                HN[i].net = aux;
                HN[i].y = sigmoid(aux);
            }
            aux = 0.0f;
            float[] yy = new float[ON.Length];
            for (int j = 0; j < ON.Length; j++)
            {
                for (int i = 0; i < ON[j].w.Length; i++)
                {
                    aux += HN[i].y * ON[j].w[i];
                }
                aux += ON[j].bias;
                ON[j].net = aux;
                ON[j].y = sigmoid(aux);
                yy[j] = ON[j].y;
            }
            return yy;
        }
        public float sigmoid(float xx)
        {
            return (1.0f / (1.0f + (float)Math.Exp(-xx)));
        }
        public class Neural
        {
            public float[] w;
            public float bias;
            public float Error;
            public float net;
            public float y;
            static Random rand = new Random();

            public Neural(int inputs)
            {
                w = new float[inputs];
                for (int i = 0; i < inputs; i++)
                {
                    w[i] = ((float)rand.Next(-1000, 1000) / 2000.0f);
                }
                bias = ((float)rand.Next(0, 2000) / 2000.0f);
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
        public void CalcDistance(float[] inputs)
        {
            for (int i = 0; i < dimension.Y; i++)
            {
                for (int j = 0; j < dimension.X; j++)
                {
                    nodes[j, i].Distance = nodes[j, i].Distance_Clac(inputs);
                    float WRow = 0.0f;
                    for (int k = 0; k < nodes[j, i].Weights.Length; k++)
                    {
                        WRow += nodes[j, i].Weights[k];
                    }

                    nodes[j, i].color = Color.FromArgb((int)(180 * WRow), (int)(180 * WRow), (int)(180 * WRow));
                }
            }
        }        
        public SOM_Nodes Get_BMU()
        {
            SOM_Nodes node = nodes[0, 0];
            float min_dis = nodes[0, 0].Distance;
            for (int i = 0; i < dimension.Y; i++)
            {
                for (int j = 0; j < dimension.X; j++)
                {
                    if(nodes[j, i].Distance < min_dis)
                    {
                        min_dis = nodes[j, i].Distance;
                        node = nodes[j, i];
                    }
                }
            }
            return node;
        }
        public void Update(List<Point> neighboring, float[] inputs, float LR)
        {
            for (int i = 0; i < neighboring.Count; i++)
            {
                for (int j = 0 ; j < inputs.Length;j++)
                {
                    nodes[neighboring[i].X, neighboring[i].Y].Weights[j] = nodes[neighboring[i].X, neighboring[i].Y].Weights[j] + LR * (inputs[j] - nodes[neighboring[i].X, neighboring[i].Y].Weights[j]);
                }
            }
        }
        public List<Point> Get_neighboring(SOM_Nodes BMU,int radius)
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
                        XX = dimension.X -1;
                    if (XX < 0)
                        XX = 0;
                    if (YY >= this.dimension.Y)
                        YY = dimension.Y -1;
                    if (YY < 0)
                        YY = 0;
                    aux = new Point(XX, YY);
                    if(!FindInList(pp,aux))
                        pp.Add(aux);
                }
            }
            return pp;
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
                    w[i] = (float)rand.NextDouble() / 2.0f;
                }
            }
            public float Distance_Clac(float[] inputs)
            {
                float dis = 0.0f;
                for (int i = 0; i < inputs.Length; i++)
                {
                    dis = ((inputs[i] - this.w[i]) * (inputs[i] - this.w[i]));
                }
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
