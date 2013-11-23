/*
 * This file contains C#/.NET 4.5 implementations for 7th week homework of the CS1156x "Learning From Data" at eDX
 * 
 * External libraries required: 
 *    math.numerics-2.6.2 (Install-Package MathNet.Numerics)
 *    libsvm.net 2.0.3    (Install-Package libsvm.net)
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
using System.Net;
using System.Text.RegularExpressions;
using libsvm;
using System.IO;

namespace StochasticTinker.edX.CS1156x.HW7
{
  /// <summary>
  ///   This class helps to supress libsvm console output
  /// </summary>
  class SvmSilencer : svm_print_interface
  {
    public void print(string str)
    {
      return;
    }
  }

  class HW7
  {
    static double[][] trainingData = Regex.Split(new WebClient().DownloadString("http://work.caltech.edu/data/in.dta"), "\r\n|\r|\n").Where(
        line => !string.IsNullOrWhiteSpace(line)).Select(
        line => line.Trim().Replace("   ", ",").Replace("  ", ",").Split(',').Select(v => Convert.ToDouble(v)).ToArray()).ToArray();

    static double[][] testData = Regex.Split(new WebClient().DownloadString("http://work.caltech.edu/data/out.dta"), "\r\n|\r|\n").Where(
        line => !string.IsNullOrWhiteSpace(line)).Select(
        line => line.Trim().Replace("   ", ",").Replace("  ", ",").Split(',').Select(v => Convert.ToDouble(v)).ToArray()).ToArray();

    static HW7()
    {
      //Supress svmlib console output
      svm.svm_set_print_string_function(new SvmSilencer());
    }

    static void Main(string[] args)
    {
      RunQ1Simulation();
      RunQ2Simulation();
      RunQ3Simulation();
      RunQ4Simulation();
      RunQ5Simulation(); 

      RunQ6Simulation();

      RunQ8Simulation();
      RunQ9_10Simulation();
    }

    /// <summary>
    /// Non-linear-transformed linear regression with validation for homework Q1 of the 
    ///  7th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ1Simulation()
    {
      var results = Enumerable.Range(3, 5).Select(k =>
        RunValidation(trainingData.Take(25).ToArray(),
        trainingData.Skip(25).ToArray(),
        testData, 
        k)).ToArray();

      var bestValidation = results.OrderBy(r => r.Item2).First();

      Console.Out.WriteLine("HW7 Q1:");
      Console.Out.WriteLine("\tk = {0}", bestValidation.Item1);
      Console.Out.WriteLine("\teVal = {0}", bestValidation.Item2);
    }

    /// <summary>
    /// Non-linear-transformed linear regression with validation for homework Q2 of the 
    ///  7th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ2Simulation()
    {
      var results = Enumerable.Range(3, 5).Select(k =>
        RunValidation(trainingData.Take(25).ToArray(),
        trainingData.Skip(25).ToArray(),
        testData,
        k)).ToArray();

      var bestValidation = results.OrderBy(r => r.Item3).First();

      Console.Out.WriteLine("HW7 Q2:");
      Console.Out.WriteLine("\tk = {0}", bestValidation.Item1);
      Console.Out.WriteLine("\teVal = {0}", bestValidation.Item2);
      Console.Out.WriteLine("\teOut = {0}", bestValidation.Item3);
    }

    /// <summary>
    /// Non-linear-transformed linear regression with validation for homework Q3 of the 
    ///  7th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ3Simulation()
    {
      var results = Enumerable.Range(3, 5).Select(k =>
        RunValidation(trainingData.Skip(25).ToArray(),
        trainingData.Take(25).ToArray(),
        testData,
        k)).ToArray();

      var bestValidation = results.OrderBy(r => r.Item2).First();

      Console.Out.WriteLine("HW7 Q3:");
      Console.Out.WriteLine("\tk = {0}", bestValidation.Item1);
      Console.Out.WriteLine("\teVal = {0}", bestValidation.Item2);
    }

    /// <summary>
    /// Non-linear-transformed linear regression with validation for homework Q4 of the 
    ///  7th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ4Simulation()
    {
      var results = Enumerable.Range(3, 5).Select(k =>
        RunValidation(trainingData.Skip(25).ToArray(),
        trainingData.Take(25).ToArray(),
        testData,
        k)).ToArray();

      var bestValidation = results.OrderBy(r => r.Item3).First();

      Console.Out.WriteLine("HW7 Q4:");
      Console.Out.WriteLine("\tk = {0}", bestValidation.Item1);
      Console.Out.WriteLine("\teVal = {0}", bestValidation.Item2);
      Console.Out.WriteLine("\teOut = {0}", bestValidation.Item3);
    }

    /// <summary>
    /// Non-linear-transformed linear regression with validation for homework Q5 of the 
    ///  7th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ5Simulation()
    {
      var results = Enumerable.Range(3, 5).Select(k =>
        RunValidation(trainingData.Take(25).ToArray(),
        trainingData.Skip(25).ToArray(),
        testData,
        k)).ToArray();

      var best1 = results.OrderBy(r => r.Item3).First();

      results = Enumerable.Range(3, 5).Select(k =>
        RunValidation(trainingData.Skip(25).ToArray(),
        trainingData.Take(25).ToArray(),
        testData,
        k)).ToArray();
      var best2 = results.OrderBy(r => r.Item3).First();

      Console.Out.WriteLine("HW7 Q5:");
      Console.Out.WriteLine("\teOut1 = {0}", best1.Item3);
      Console.Out.WriteLine("\teOut2 = {0}", best2.Item3);
    }

    /// <summary>
    /// Calculation (via simulation) of the expected value of the minimum of two random uniformly distributed 
    /// values, i.e. E(min(e1, e2)) for homework Q6 of the 7th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ6Simulation()
    {
      Random rnd1 = new Random(12345);
      Random rnd2 = new Random(78910);

      const int N = 100000;

      double sum = 0;
      for (int i = 0; i < N; i++) 
        sum += Math.Min(rnd1.NextDouble(), rnd2.NextDouble());

      Console.Out.WriteLine("HW7 Q6:");
      Console.Out.WriteLine("\tE(e) = {0}", sum / N);
    }

    static Tuple<int, double, double> RunValidation(double[][] trainingData, double[][] validationData, double[][] outSampleData, int k)
    {
      int N = trainingData.Length;
      var ZZ = new DenseMatrix(N, 8);
      var Y = new DenseVector(N);
      for (int j = 0; j < N; j++)
      {
        double x1 = trainingData[j][0], x2 = trainingData[j][1];
        ZZ[j, 0] = 1;
        ZZ[j, 1] = x1;
        ZZ[j, 2] = x2;
        ZZ[j, 3] = x1 * x1;
        ZZ[j, 4] = x2 * x2;
        ZZ[j, 5] = x1 * x2;
        ZZ[j, 6] = Math.Abs(x1 - x2);
        ZZ[j, 7] = Math.Abs(x1 + x2);
         
        Y[j] = trainingData[j][2];
      }

      var Z = ZZ.SubMatrix(0, N, 0, k + 1);
      var WW = Z.TransposeThisAndMultiply(Z).Inverse().TransposeAndMultiply(Z).Multiply(Y);

      var W = new DenseVector(8);
      WW.CopySubVectorTo(W, 0, 0, k + 1);

      Func<double, double, double> h = (x1, x2) =>
        W[0] + W[1] * x1 + W[2] * x2 + W[3] * x1 * x1 + W[4] * x2 * x2 + W[5] * x1 * x2
        + W[6] * Math.Abs(x1 - x2) + W[7] * Math.Abs(x1 + x2) >= 0 ? 1.0 : -1.0;

      return Tuple.Create(
        k,
        (validationData.Count(v => Math.Sign(h(v[0], v[1])) != Math.Sign(v[2])) + 0.0) / validationData.Length, 
        (outSampleData.Count(v => Math.Sign(h(v[0], v[1])) != Math.Sign(v[2])) + 0.0) / outSampleData.Length);
    }

    /// <summary>
    /// PLA vs SVM simulation for homework Q8 of the 7th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ8Simulation()
    {
      var results = RunPLAvsSVM(1000, 10);

      Console.Out.WriteLine("HW7 Q8:");
      Console.Out.WriteLine("\tSVM wins = {0}", results.Item1);
    }

    /// <summary>
    /// PLA vs SVM simulation for homework Q9 & Q10 of the 7th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ9_10Simulation()
    {
      var results = RunPLAvsSVM(1000, 100);

      Console.Out.WriteLine("HW7 Q9:");
      Console.Out.WriteLine("\tSVM wins = {0}", results.Item1);

      Console.Out.WriteLine("HW7 Q10:");
      Console.Out.WriteLine("\t# of SVs = {0}", results.Item2);
    }

    static Tuple<double, double> RunPLAvsSVM(int experiments, int points)
    {
      const int TEST_POINTS = 10000;
      Random rnd = new Random();

      long svmWins = 0, svCount = 0;
      for (int i = 1; i <= experiments; i++)
      {
        //pick a random line y = a * x + b
        double x1 = rnd.NextDouble(), y1 = rnd.NextDouble(), x2 = rnd.NextDouble(), y2 = rnd.NextDouble();
        var Wf = new DenseVector(3);
        Wf[0] = 1;
        Wf[1] = (y1 - y2) / (x1 * y2 - y1 * x2);
        Wf[2] = (x2 - x1) / (x1 * y2 - y1 * x2);
        Func<MathNet.Numerics.LinearAlgebra.Generic.Vector<double>, int> f = x => Wf.DotProduct(x) >= 0 ? 1 : -1;

        //generate training set of N random points
        var X = new DenseMatrix(points, 3);
        do
          for (int j = 0; j < points; j++)
          {
            X[j, 0] = 1;
            X[j, 1] = rnd.NextDouble() * 2 - 1;
            X[j, 2] = rnd.NextDouble() * 2 - 1;
          }
        while (Enumerable.Range(0, X.RowCount).All(j => f(X.Row(0)) == f(X.Row(j))));
   
        var W = new DenseVector(3);
        Func<MathNet.Numerics.LinearAlgebra.Generic.Vector<double>, int> h = x => W.DotProduct(x) >= 0 ? 1 : -1; 

        //run Perceptron
        int k = 1;
        while (Enumerable.Range(0, points).Any(j => h(X.Row(j)) != f(X.Row(j))))
        {
          //find all misclasified points
          int[] M = Enumerable.Range(0, points).Where(j => h(X.Row(j)) != f(X.Row(j))).ToArray();
          int m = M[rnd.Next(0, M.Length)];

          int sign = f(X.Row(m));
          W[0] += sign;
          W[1] += sign * X[m, 1];
          W[2] += sign * X[m, 2];
          k++;
        }

        //calculate P[f(Xtest) != h(Xtest)]
        DenseVector Xtest = new DenseVector(3);
        Xtest[0] = 1;
        int matches = 0;
        for (int j = 0; j < TEST_POINTS; j++)
        {
          Xtest[1] = rnd.NextDouble() * 2 - 1;
          Xtest[2] = rnd.NextDouble() * 2 - 1;
          if (f(Xtest) == h(Xtest)) matches++;
        }
        double Ppla = (matches + 0.0) / TEST_POINTS;

        //Run SVM
        var prob = new svm_problem() {
          x = Enumerable.Range(0, points).Select(j => 
            new svm_node[] { 
              new svm_node() { index = 0, value = X[j, 1] }, 
              new svm_node() { index = 1, value = X[j, 2] } }).ToArray(),
          y = Enumerable.Range(0, points).Select(j => (double)f(X.Row(j))).ToArray(),
          l = points };
        
        var model = svm.svm_train(prob, new svm_parameter()
        {
          svm_type = (int)SvmType.C_SVC,
          kernel_type = (int)KernelType.LINEAR,
          C = 1000000,
          eps = 0.001,
          shrinking = 0
        });

        //calculate P[f(Xtest) != h_svm(Xtest)]
        svm_node[] Xsvm = new svm_node[] { 
              new svm_node() { index = 0, value = 1.0 }, 
              new svm_node() { index = 1, value = 1.0 } };
        matches = 0;

        for (int j = 0; j < TEST_POINTS; j++)
        {
          Xtest[1] = rnd.NextDouble() * 2 - 1;
          Xsvm[0].value = Xtest[1];
          Xtest[2] = rnd.NextDouble() * 2 - 1;
          Xsvm[1].value = Xtest[2];
          if (f(Xtest) == (svm.svm_predict(model, Xsvm) > 0 ? 1 : -1)) matches++;
        }
        double Psvm = (matches + 0.0) / TEST_POINTS;

        svCount += model.l;
        if (Psvm >= Ppla) svmWins++;
      }

      return Tuple.Create((svmWins + 0.0) / experiments, (svCount + 0.0) / experiments);
    }

  }
}
