using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimplePerceptron
{
    class Program
    {
        //Taxa de aprendizado
        private static double n = 0.2;
        static void Main(string[] args)
        {
            // Entradas
            List<int[]> inputs = new List<int[]>(2);
            inputs.Add(new int[] { 
                0, 1, 1, 0, 
                1, 0, 0, 1, 
                1, 0, 0, 1, 
                0, 1, 1, 0 });

            inputs.Add(new int[] { 
                1, 0, 0, 1, 
                1, 0, 0, 1, 
                1, 1, 1, 1, 
                1, 0, 0, 1 });

            // Saídas 
            List<int> outputs = new List<int>(2);
            outputs.Add(0);
            outputs.Add(1);

            // Pesos (sinapses)
            List<double[]> weights = new List<double[]>(1);
            weights.Add(new double[17]);

            Random random = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < weights[0].Length; i++) {
                weights[0][i] = random.NextDouble();
            }

            double erro = 1;

            while (erro > 0) { 
                // Calculando a saída
                double[] y = EvaluateOutput(inputs, weights[0]);
                for(int i = 0; i < y.Length; i ++){
                    Console.WriteLine("VK: " + i.ToString() + ": " + y[i].ToString());
                }

                // Aplicando funcao de ativacao
                ActivationFunction(y);
                for (int i = 0; i < y.Length; i++){
                    Console.WriteLine("VK: " + i.ToString() + ": " + y[i].ToString());
                }

                // Calculo do erro - Para todas as entradas, a Soma dos erros (entre o desejado e obtido)
                erro = 0;
                for (int i = 0; i < y.Length; i++)
                {
                    // Precisa ser o valor absoluto (pq?) pq se tiver um valor -1 e 1, vai dar erro 0;
                    erro += Math.Abs(outputs[i] - y[i]);
                }

                // Ajustando os pesos - "Passamos um peso, vc atualiza os pesos de UM NEURONIO" - ZAN, Caver
                UpdateWeights(inputs, outputs, y, weights[0]);

            }

            //Testando com um padrao parecido
            // Entradas
            List<int[]> tests = new List<int[]>(2);
            tests.Add(new int[] { 
                0, 1, 1, 0, 
                0, 0, 0, 1, 
                1, 0, 0, 1, 
                0, 1, 1, 0 });

            tests.Add(new int[] { 
                1, 0, 0, 1, 
                1, 0, 0, 1, 
                1, 0, 1, 1, 
                1, 0, 0, 1 });


            Console.WriteLine("\n\nTeste\n");
            double[] s = EvaluateOutput(tests, weights[0]);
            for (int i = 0; i < s.Length; i++)
            {
                Console.WriteLine("VK: " + i.ToString() + ": " + s[i].ToString());
            }

            // Esperando o usuario sair
            Console.ReadKey();

        }

        // Calcula o potencial de ativacao {VK} do neuronio para as varias entradas
        private static double[] EvaluateOutput(List<int[]> inputs, double[] weights)
        {
            double[] y = new double[inputs.Count];

            // Percorrendo entradas
            for (int i = 0; i < inputs.Count; i++) {
                // Somatoria
                y[i] = 0;
                for (int j = 0; j < inputs[i].Length; j++) {
                    // Aqui esta o VK, o potencial de ativacao
                    y[i] += inputs[i][j] * weights[j];  
                }

                // Bias
                y[i] += weights[weights.Length - 1];
            }


            return y;
        }

        // Funcao de ativacao - Serve para evitar que um sinal muito alto influencie toda a rede neural
        private static void ActivationFunction(double[] y)
        {
            for (int i = 0; i < y.Length; i++)
            {
                y[i] = (y[i] > 0 ? 1 : 0);
            }
        }

        // Atualiza os pesso de UM neuronio -- > saidas desejadas (outputs), saidas obitdas(y)
        private static void UpdateWeights(List<int[]> inputs, List<int> outputs, double[] y, double[] w)
        {
            // Percorre entradas
            for (int i = 0; i < inputs.Count; i++)
            {
                // Erro = entrada desejada - entrada obtida
                double e = outputs[i] - y[i];

                // Percorre o valor da entrada
                for (int j = 0; j < inputs[i].Length; j++)
                {
                    //dW = entrada (inputs) * taxa aprendizado (n) * erro (e) 
                    double deltaW = inputs[i][j] * n * e;
                    //Atualiza o peso
                    w[j] += deltaW;
                }

                //Bias = entrada (inputs) * erro (e) 
                w[w.Length - 1] += n * outputs[i] - y[i];
            }
        }

    }
}
