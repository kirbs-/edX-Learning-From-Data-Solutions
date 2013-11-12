/*
 * This file contains C#/.NET 4.5 implementations for 6th week homework of the CS1156x "Learning From Data" at eDX
 * 
 * External library required: math.numerics-2.6.2 (Install-Package MathNet.Numerics)
 * 
 * Author: stochastictinker
 * Nov 2013
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Double;

namespace StochasticTinker.edX.CS1156x.HW6
{
  class HW6
  {
    static void Main(string[] args)
    {
      RunQ2Simulation();
      RunQ3Simulation();
      RunQ4Simulation();
      RunQ5Simulation();
      RunQ6Simulation();
    }

    /// <summary>
    /// Non-linear-transformed linear regression simulation for homework Q2 of the 
    ///  6th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ2Simulation()
    {
      //load training set
      var trainingData = System.IO.File.ReadLines(@"c:/projects/studies/edx/cs1156x/net/hw6/in.dta").Select(
        line => line.Trim().Replace("   ", ",").Replace("  ", ",").Split(',').Select(v => Convert.ToDouble(v)).ToArray()).ToArray();

      //load test set
      var testData = System.IO.File.ReadLines(@"c:/projects/studies/edx/cs1156x/net/hw6/out.dta").Select(
        line => line.Trim().Replace("   ", ",").Replace("  ", ",").Split(',').Select(v => Convert.ToDouble(v)).ToArray()).ToArray();

      int N = trainingData.Length;
      var Z = new DenseMatrix(N, 8);
      var Y = new DenseVector(N);
      for (int j = 0; j < N; j++)
      {
        double x1 = trainingData[j][0], x2 = trainingData[j][1]; 
        Z[j, 0] = 1;
        Z[j, 1] = x1;
        Z[j, 2] = x2;
        Z[j, 3] = x1 * x1;
        Z[j, 4] = x2 * x2;
        Z[j, 5] = x1 * x2;
        Z[j, 6] = Math.Abs(x1 - x2);
        Z[j, 7] = Math.Abs(x1 + x2);

        Y[j] = trainingData[j][2];
      }

      //Z.QR().Solve(DenseMatrix.Identity(Z.RowCount)).Multiply(Y);
      var W = Z.TransposeThisAndMultiply(Z).Inverse().TransposeAndMultiply(Z).Multiply(Y);

      Func<double, double, double> h = (x1, x2) =>
        W[0] + W[1] * x1 + W[2] * x2 + W[3] * x1 * x1 + W[4] * x2 * x2 + W[5] * x1 * x2
        + W[6] * Math.Abs(x1 - x2) + W[7] * Math.Abs(x1 + x2) >= 0 ? 1.0 : -1.0;

      double eIn = (trainingData.Count(v => Math.Sign(h(v[0], v[1])) != Math.Sign(v[2])) + 0.0) / trainingData.Length;
      double eOut = (testData.Count(v => Math.Sign(h(v[0], v[1])) != Math.Sign(v[2])) + 0.0) / testData.Length;
      
      Console.Out.WriteLine("HW6 Q2:");
      Console.Out.WriteLine("\teIn = {0}", eIn);
      Console.Out.WriteLine("\teOut = {0}", eOut);
    }

    /// <summary>
    /// Non-linear-transformed linear regression with weight decay regularizer simulation for homework Q3 of the 
    ///  6th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ3Simulation()
    {
      var e = Q3_6Simulation(-3);

      Console.Out.WriteLine("HW6 Q3:");
      Console.Out.WriteLine("\teIn = {0}", e.Item1);
      Console.Out.WriteLine("\teOut = {0}", e.Item2);
    }

    /// <summary>
    /// Non-linear-transformed linear regression with weight decay regularizer simulation for homework Q4 of the 
    ///  6th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ4Simulation()
    {
      var e = Q3_6Simulation(3);

      Console.Out.WriteLine("HW6 Q4:");
      Console.Out.WriteLine("\teIn = {0}", e.Item1);
      Console.Out.WriteLine("\teOut = {0}", e.Item2);
    }

    /// <summary>
    /// Non-linear-transformed linear regression with weight decay regularizer simulation for homework Q5 of the 
    ///  6th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ5Simulation()
    {
      var e = Q3_6Simulation(3);

      Console.Out.WriteLine("HW6 Q5:");
      Console.Out.WriteLine("\teIn = {0}", Enumerable.Range(-2, 5).OrderBy(k => Q3_6Simulation(k).Item2).First());
    }

    /// <summary>
    /// Non-linear-transformed linear regression with weight decay regularizer simulation for homework Q6 of the 
    ///  6th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ6Simulation()
    {
      var e = Q3_6Simulation(3);

      Console.Out.WriteLine("HW6 Q5:");
      Console.Out.WriteLine("\teIn = {0}", Enumerable.Range(-100, 200).Select(k => Q3_6Simulation(k).Item2).Min());
    }


    static private Tuple<double, double> Q3_6Simulation(int k)
    {
      double lambda = Math.Pow(10, k);

      //load training set
      var trainingData = System.IO.File.ReadLines(@"c:/projects/studies/edx/cs1156x/net/hw6/in.dta").Select(
        line => line.Trim().Replace("   ", ",").Replace("  ", ",").Split(',').Select(v => Convert.ToDouble(v)).ToArray()).ToArray();

      //load test set
      var testData = System.IO.File.ReadLines(@"c:/projects/studies/edx/cs1156x/net/hw6/out.dta").Select(
        line => line.Trim().Replace("   ", ",").Replace("  ", ",").Split(',').Select(v => Convert.ToDouble(v)).ToArray()).ToArray();

      int N = trainingData.Length;
      var Z = new DenseMatrix(N, 8);
      var Y = new DenseVector(N);
      for (int j = 0; j < N; j++)
      {
        double x1 = trainingData[j][0], x2 = trainingData[j][1];
        Z[j, 0] = 1;
        Z[j, 1] = x1;
        Z[j, 2] = x2;
        Z[j, 3] = x1 * x1;
        Z[j, 4] = x2 * x2;
        Z[j, 5] = x1 * x2;
        Z[j, 6] = Math.Abs(x1 - x2);
        Z[j, 7] = Math.Abs(x1 + x2);

        Y[j] = trainingData[j][2];
      }

      var W = Z.TransposeThisAndMultiply(Z).Add(DenseMatrix.Identity(8).Multiply(lambda)).Inverse().TransposeAndMultiply(Z).Multiply(Y);

      Func<double, double, double> h = (x1, x2) =>
        W[0] + W[1] * x1 + W[2] * x2 + W[3] * x1 * x1 + W[4] * x2 * x2 + W[5] * x1 * x2
        + W[6] * Math.Abs(x1 - x2) + W[7] * Math.Abs(x1 + x2) >= 0 ? 1.0 : -1.0;

      double eIn = (trainingData.Count(v => Math.Sign(h(v[0], v[1])) != Math.Sign(v[2])) + 0.0) / trainingData.Length;
      double eOut = (testData.Count(v => Math.Sign(h(v[0], v[1])) != Math.Sign(v[2])) + 0.0) / testData.Length;

      return Tuple.Create(eIn, eOut);
    }

  }
}
