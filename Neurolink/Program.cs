using System;

namespace Neurolink
{
    class Program
    {
        static void Main(string[] args)
        {
            // массив входных обучающих векторов
            Vector[] X = {
                new Vector(0, 0),
                new Vector(0, 1),
                new Vector(1, 0),
                new Vector(1, 1)
            };

            // массив выходных обучающих векторов
            Vector[] Y = {
                new Vector(0.0), // 0 ^ 0 = 0
                new Vector(1.0), // 0 ^ 1 = 1
                new Vector(1.0), // 1 ^ 0 = 1
                new Vector(0.0) // 1 ^ 1 = 0
            };

            Network network = new Network(new int[] { 2, 3, 1 });

            network.Train(X, Y, 0.5, 1e-7, 100000); // запускаем обучение сети 

            for (int i = 0; i < 4; i++)
            {
                Vector output = network.Forward(X[i]);
                Console.WriteLine("X: {0} {1}, Y: {2}, output: {3}", X[i][0], X[i][1], Y[i][0], output[0]);
            }
            Console.ReadLine();
        }
    }
}
