/*
 * This file contains C#/.NET 4.x implementations for 2nd week homework of the CS1156x "Learning From Data" at eDX
 * The code is quite dirty as it was hastily developed and never ment to be published. Have fun :)
 * 
 * External library required: math.numerics-2.6.1.30
 * 
 * Author: stochastictinker
 * Oct 2013
 */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra.Double;

namespace StochasticTinker.edX.CS1156x.HW2
{
  class HW2
  {
    static void Main(string[] args)
    {
      RunQ1Simulation();
      RunQ5Q6Simulation();
      RunQ7Simulation();
      RunQ8Simulation();
      RunQ9Q10Simulation();
    }

    /// <summary>
    /// Coin-flipping simulation for  for homework Q1 of the 
    ///  2nd week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ1Simulation()
    {
      const int EXPERIMENT_COUNT = 100000, N = 1000 , M = 10;
      Random rnd = new Random();

      double avgV1 = 0, avgVRand = 0, avgVMin = 0;
      for (int i = 1; i <= EXPERIMENT_COUNT; i++)
      {
        //flip coins
        int[] heads = new int[N];
        for (int j = 0; j < N; j++)
          for (int k = 0; k < M; k++)
            heads[j] += rnd.Next(0, 2);

        double v1 = (heads[0] + 0.0) / M;

        int jRand = rnd.Next(0, N);
        double vRand = (heads[jRand] + 0.0) / M;

        int jMin = 0, hMin = 10;
        for (int j = 0; j < N && hMin != 0; j++)
        {
          if (heads[j] < hMin)
          {
            hMin = heads[j];
            jMin = j;
          }
        }
        double vMin = (hMin + 0.0) / M;

        avgV1 += v1;
        avgVRand += vRand;
        avgVMin += vMin;
      }

      Console.Out.WriteLine("HW2 Q1:");
      Console.Out.WriteLine("\tV1 = {0}", avgV1 / EXPERIMENT_COUNT);
      Console.Out.WriteLine("\tVrand = {0}", avgVRand / EXPERIMENT_COUNT);
      Console.Out.WriteLine("\tVmin = {0}", avgVMin / EXPERIMENT_COUNT);
    }

    /// <summary>
    /// Linear regression simulation for homework Q5-Q6 of the 
    ///  2nd week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ5Q6Simulation()
    {
      const int EXPERIMENT_COUNT = 1000, N = 100;
      Random rnd = new Random();

      double avgEin = 0, avgEout = 0;
      for (int i = 1; i <= EXPERIMENT_COUNT; i++)
      {
        //pick a random line y = a1 * x + b1
        double x1 = rnd.NextDouble(), y1 = rnd.NextDouble(), x2 = rnd.NextDouble(), y2 = rnd.NextDouble();
        double a = (y1 - y2) / (x1 - x2), b = y1 - a * x1;
        Func<double, double, int> f = (x, y) => a * x + b >= y ? 1 : -1;

        //generate training set of N random points
        var X = new DenseMatrix(N, 3);
        var Y = new DenseVector(N);
        for (int j = 0; j < N; j++)
        {
          X[j, 0] = 1;
          X[j, 1] = rnd.NextDouble() * 2 - 1;
          X[j, 2] = rnd.NextDouble() * 2 - 1;

          Y[j] = f(X[j, 1], X[j, 2]);
        }

        var W = X.QR().Solve(DenseMatrix.Identity(X.RowCount)).Multiply(Y);

        Func<double, double, int> h = (x, y) => W[0] + W[1] * x + W[2] * y >= 0 ? 1 : -1;

        //find Ein
        int count = 0;
        for (int j = 0; j < N; j++) if (h(X[j, 1], X[j, 2]) != Y[j]) count++;
        avgEin += (count + 0.0) / N;

        //find p: f != g
        const int P_SAMPLE_COUNT = 1000;
        count = 0;
        for (int j = 1; j <= P_SAMPLE_COUNT; j++)
        {
          double xx = rnd.NextDouble() * 2 - 1;
          double yy = rnd. NextDouble() * 2 - 1;
          if (f(xx, yy) != h(xx, yy)) count++;
        }

        avgEout += (count + 0.0) / P_SAMPLE_COUNT;
      }

      Console.Out.WriteLine("HW2 Q5:");
      Console.Out.WriteLine("\tEin = {0}", avgEin / EXPERIMENT_COUNT);
      Console.Out.WriteLine("HW2 Q6:");
      Console.Out.WriteLine("\tEout = {0}", avgEout / EXPERIMENT_COUNT);
    }

    /// <summary>
    /// Linear regression/Perceptron simulation for homework Q7 of the 
    ///  2nd week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ7Simulation()
    {
      const int EXPERIMENT_COUNT = 1000, N = 10;
      Random rnd = new Random();

      double avgK = 0;
      for (int i = 1; i <= EXPERIMENT_COUNT; i++)
      {
        //pick a random line y = a1 * x + b1
        double x1 = rnd.NextDouble(), y1 = rnd.NextDouble(), x2 = rnd.NextDouble(), y2 = rnd.NextDouble();
        double a = (y1 - y2) / (x1 - x2), b = y1 - a * x1;
        Func<double, double, int> f = (x, y) => a * x + b >= y ? 1 : -1;

        //generate training set of N random points
        var X = new DenseMatrix(N, 3);
        var Y = new DenseVector(N);
        for (int j = 0; j < N; j++)
        {
          X[j, 0] = 1;
          X[j, 1] = rnd.NextDouble() * 2 - 1;
          X[j, 2] = rnd.NextDouble() * 2 - 1;

          Y[j] = f(X[j, 1], X[j, 2]);
        }

        var W = X.QR().Solve(DenseMatrix.Identity(X.RowCount)).Multiply(Y);

        double w0 = W[0], w1 = W[1], w2 = W[2];

        Func<double, double, int> h = (x, y) => w0 + w1 * x + w2 * y >= 0 ? 1 : -1;

        //run Perceptron
        int k = 1;
        while (Enumerable.Range(0, N).Any(j => f(X[j, 1], X[j, 2]) != h(X[j, 1], X[j, 2])))
        {
          //find all misclasified points
          int[] M = Enumerable.Range(0, N).Where(j => f(X[j, 1], X[j, 2]) != h(X[j, 1], X[j, 2])).ToArray();
          int m = M[rnd.Next(0, M.Length)];

          int sign = f(X[m, 1], X[m, 2]);
          w0 += sign;
          w1 += sign * X[m, 1];
          w2 += sign * X[m, 2];
          k++;
        }

        avgK += k;
      }

      Console.Out.WriteLine("HW2 Q7:");
      Console.Out.WriteLine("\tK = {0}", avgK / EXPERIMENT_COUNT);
    }

    /// <summary>
    /// Linear regressionsimulation with non-separable target function
    /// for homework Q8 of the 2nd week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ8Simulation()
    {
      const int EXPERIMENT_COUNT = 1000, N = 100;
      Random rnd = new Random();

      double avgEin = 0;
      for (int i = 1; i <= EXPERIMENT_COUNT; i++)
      {
        Func<double, double, int> f = (x1, x2) => x1 * x1 + x2 * x2 - 0.6 >= 0 ? 1 : -1;

        //generate training set of N random points
        var X = new DenseMatrix(N, 3);
        var Y = new DenseVector(N);
        for (int j = 0; j < N; j++)
        {
          X[j, 0] = 1;
          X[j, 1] = rnd.NextDouble() * 2 - 1;
          X[j, 2] = rnd.NextDouble() * 2 - 1;

          Y[j] = f(X[j, 1], X[j, 2]);
          
          //not exactly how it was defined in the problem statement, but shall be good enough
          if (rnd.NextDouble() < 0.1) Y[j] = -Y[j];
        }

        var W = X.QR().Solve(DenseMatrix.Identity(X.RowCount)).Multiply(Y);

        Func<double, double, int> h = (x, y) => W[0] + W[1] * x + W[2] * y >= 0 ? 1 : -1;

        //find Ein
        int count = 0;
        for (int j = 0; j < N; j++) if (h(X[j, 1], X[j, 2]) != Y[j]) count++;
        avgEin += (count + 0.0) / N;
      }

      Console.Out.WriteLine("HW2 Q8:");
      Console.Out.WriteLine("\tEin = {0}", avgEin / EXPERIMENT_COUNT);
    }

    /// <summary>
    /// Non-linear-transformed linear regression simulation for homework Q9, Q10 of the 
    ///  2nd week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ9Q10Simulation()
    {
      const int EXPERIMENT_COUNT = 1000, N = 100;
      Random rnd = new Random();

      double avgEout = 0;
      for (int i = 1; i <= EXPERIMENT_COUNT; i++)
      {
        Func<double, double, int> f = (x1, x2) => x1 * x1 + x2 * x2 - 0.6 >= 0 ? 1 : -1;

        //generate training set of N random points
        var X = new DenseMatrix(N, 3);
        var Y = new DenseVector(N);
        for (int j = 0; j < N; j++)
        {
          X[j, 0] = 1;
          X[j, 1] = rnd.NextDouble() * 2 - 1;
          X[j, 2] = rnd.NextDouble() * 2 - 1;

          Y[j] = f(X[j, 1], X[j, 2]);

          // Just flipping each Y with a 10% chance - 
          // not exactly how it was defined in the problem statement, but shall be good enough
          if (rnd.NextDouble() < 0.1) Y[j] = -Y[j];
        }

        var XX = new DenseMatrix(N, 6);
        for (int j = 0; j < N; j++)
        {
          XX[j, 0] = 1;
          XX[j, 1] = X[j, 1];
          XX[j, 2] = X[j, 2];
          XX[j, 3] = X[j, 1] * X[j, 2];
          XX[j, 4] = X[j, 1] * X[j, 1];
          XX[j, 5] = X[j, 2] * X[j, 2];
        }

        var W = XX.QR().Solve(DenseMatrix.Identity(XX.RowCount)).Multiply(Y);

        Func<double, double, int> h = (x, y) => W[0] + W[1] * x + W[2] * y + W[3] * x * y + W[4] * x * x + W[5] * y * y >= 0 ? 1 : -1;

        //find p: f != g
        const int P_SAMPLE_COUNT = 1000;
        int count = 0;
        for (int j = 1; j <= P_SAMPLE_COUNT; j++)
        {
          double xx = rnd.NextDouble() * 2 - 1;
          double yy = rnd.NextDouble() * 2 - 1;
          int ff = f(xx, yy);
          if (rnd.NextDouble() < 0.1) ff = -ff;

          if (ff != h(xx, yy)) count++;
        }

        avgEout += (count + 0.0) / P_SAMPLE_COUNT;
      }

      Console.Out.WriteLine("HW2 Q10:");
      Console.Out.WriteLine("\tEout = {0}", avgEout / EXPERIMENT_COUNT);
    }
  }
}
