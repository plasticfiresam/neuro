namespace Neurolink
{
    class Vector
    {
        public double[] v; // значения вектора
        public int n; // длина вектора

        // конструктор из длины
        public Vector(int n)
        {
            this.n = n; // копируем длину
            v = new double[n]; // создаём массив
        }

        // создание вектора из вещественных значений
        public Vector(params double[] values)
        {
            n = values.Length;
            v = new double[n];

            for (int i = 0; i < n; i++)
                v[i] = values[i];
        }

        // обращение по индексу
        public double this[int i]
        {
            get { return v[i]; } // получение значение
            set { v[i] = value; } // изменение значения
        }
    }
}
