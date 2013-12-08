/*
 * This file contains C#/.NET 4.5 implementations for final exam of the CS1156x "Learning From Data" at eDX
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
using System.Net;
using System.Text.RegularExpressions;
using MathNet.Numerics.LinearAlgebra.Double;
using libsvm;

namespace StochasticTinker.edX.CS1156x.Exam
{
  class Exam
  {
    static double[][] trainingData = Regex.Split(new WebClient().DownloadString("http://www.amlbook.com/data/zip/features.train"), "\r\n|\r|\n").Where(
    line => !string.IsNullOrWhiteSpace(line)).Select(
    line => line.Trim().Replace("   ", ",").Replace("  ", ",").Split(',').Select(v => Convert.ToDouble(v)).ToArray()).ToArray();

    static double[][] testData = Regex.Split(new WebClient().DownloadString("http://www.amlbook.com/data/zip/features.test"), "\r\n|\r|\n").Where(
        line => !string.IsNullOrWhiteSpace(line)).Select(
        line => line.Trim().Replace("   ", ",").Replace("  ", ",").Split(',').Select(v => Convert.ToDouble(v)).ToArray()).ToArray();

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

    static Exam()
    {
      //Supress svmlib console output
      svm.svm_set_print_string_function(new SvmSilencer());
    }

    static void Main(string[] args)
    {
      RunQ7Simulation();
      RunQ8Simulation();
      RunQ9Simulation();
      RunQ10Simulation();
      RunQ12Simulation();
      RunQ13Simulation();
      RunQ14Simulation();
      RunQ15Simulation();
      RunQ16Simulation();
      RunQ17Simulation();
      RunQ18Simulation();
    }

    /// <summary>
    /// Linear regression with weight decay regularizer simulation for final exam Q7 of the course CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ7Simulation()
    {
      var results = new[] { 5, 6, 7, 8, 9 }.Select(d => 
        {
          var stats = RunRegularizedRegression(
            Get1vsAllData(trainingData, d),
            Get1vsAllData(testData, d),
            (x1, x2) => new [] {1, x1, x2});
          return new 
          {
            digit = d,
            eIn = stats.Item1,
            eOut = stats.Item2
          };
        }).ToArray();

      Console.Out.WriteLine("Exam Q7:");
      Console.Out.WriteLine("\teIn = {0}", results.OrderBy(v => v.eIn).First().eIn);
      Console.Out.WriteLine("\tdigit = {0}", results.OrderBy(v => v.eIn).First().digit);
    }

    /// <summary>
    /// Quadratically-transformed Linear regression with weight decay regularizer simulation 
    ///   for final exam Q8 of the course CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ8Simulation()
    {
      var results = new[] { 0, 1, 2, 3, 4 }.Select(d =>
      {
        var stats = RunRegularizedRegression(
          Get1vsAllData(trainingData, d),
          Get1vsAllData(testData, d),
          (x1, x2) => new[] { 1, x1, x2, x1 * x2, x1 * x1, x2 * x2 });
        return new
        {
          digit = d,
          eIn = stats.Item1,
          eOut = stats.Item2
        };
      }).ToArray();

      Console.Out.WriteLine("Exam Q8:");
      Console.Out.WriteLine("\teOut = {0}", results.OrderBy(v => v.eOut).First().eOut);
      Console.Out.WriteLine("\tdigit = {0}", results.OrderBy(v => v.eOut).First().digit);
    }

    /// <summary>
    /// Linear regression with weight decay regularizer simulation for final exam Q9 of the course CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ9Simulation()
    {
      Console.Out.WriteLine("Exam Q9:");
      for (int d = 0; d <= 9; d++)
      {
        var lin = RunRegularizedRegression(
          Get1vsAllData(trainingData, d),
          Get1vsAllData(testData, d),
          (x1, x2) => new[] { 1, x1, x2});

        var quad = RunRegularizedRegression(
          Get1vsAllData(trainingData, d),
          Get1vsAllData(testData, d),
          (x1, x2) => new[] { 1, x1, x2, x1 * x2, x1 * x1, x2 * x2 });

        Console.Out.WriteLine("\t{0} versus all -----------------------", d);
        Console.Out.WriteLine("\toverfitted: {0}", lin.Item1 > quad.Item1 && lin.Item2 < quad.Item2);
        Console.Out.WriteLine("\tEout improved by: {0}%", -(quad.Item2 - lin.Item2) / lin.Item2 * 100);
      }
    }

    /// <summary>
    /// Linear regression with weight decay regularizer simulation for final exam Q10 of the course CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ10Simulation()
    {
      var results = new [] {1.0, 0.01}.Select(lambda => 
        {
          var stats =  RunRegularizedRegression(
            Get1vs1Data(trainingData, 1, 5),
            Get1vs1Data(testData, 1, 5),
            (x1, x2) => new[] { 1, x1, x2, x1 * x2, x1 * x1, x2 * x2 },
            lambda);
          return new {
            lambda = lambda,
            Ein = stats.Item1,
            Eout = stats.Item2
          };
        }).ToArray();

      var r_1_0 = results.First(v => v.lambda == 1.0);
      var r_0_01 = results.First(v => v.lambda == 0.01);

      Console.Out.WriteLine("Exam Q10:");
      Console.Out.WriteLine("\tEin(1.0):\t{0}", r_1_0.Ein);
      Console.Out.WriteLine("\tEout(1.0):\t{0}", r_1_0.Eout);
      Console.Out.WriteLine("\tEin(0.001):\t{0}", r_0_01.Ein);
      Console.Out.WriteLine("\tEout(0.001):\t{0}", r_0_01.Eout);
      Console.Out.WriteLine("\toverfitted:\t{0}", r_1_0.Ein > r_0_01.Ein && r_1_0.Eout < r_0_01.Eout);
    }

    /// <summary>
    /// Quadratic kernel SVM simulation for final exam Q12 of the course CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ12Simulation()
    {
      int[][] data = new [] {
        new []{ 1,  0, -1},
        new []{ 0,  1, -1},
        new []{ 0, -1, -1},
        new []{-1,  0,  1},
        new []{ 0,  2,  1},
        new []{ 0, -2,  1},
        new []{-2,  0,  1}};

      var training = new svm_problem()
      {
        x = data.Select(v =>
          new svm_node[] { 
              new svm_node() { index = 0, value = v[0] }, 
              new svm_node() { index = 1, value = v[1] } }).ToArray(),
        y = data.Select(v => (double)v[2]).ToArray(),
        l = data.Length
      };

      var model = svm.svm_train(training, new svm_parameter()
      {
        svm_type = (int)SvmType.C_SVC,
        kernel_type = (int)KernelType.POLY,
        C = 100000000000,
        degree = 2,
        coef0 = 1,
        gamma = 1,
        eps = 0.001,
        shrinking = 0
      });

      Console.Out.WriteLine("Exam Q12:");
      Console.Out.WriteLine("\t# of support vectors:\t{0}", model.l);

      Func<svm_problem, double> E = p => (p.x.Zip(p.y, (v, u) => new { x = v, y = u }).Count(v =>
        Math.Sign(svm.svm_predict(model, v.x)) != Math.Sign(v.y)) + 0.0) / p.x.Length;

      Console.Out.WriteLine("\tEin:\t{0}", E(training));
    }

    /// <summary>
    /// RBF SVM simulation for final exam Q13 of the course CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ13Simulation()
    {
      const int ITERATIONS = 1000;

      double P = 0;
      for (int i = 0; i < ITERATIONS; i++)
      {
        var data = GenerateSinDataSet();
        P += (RunSvm(data, data).Item1 != 0 ? 1 : 0);
      }

      P /= ITERATIONS;

      Console.Out.WriteLine("Exam Q13:");
      Console.Out.WriteLine("\t% not separable: {0}", P * 100);
    }

    /// <summary>
    /// Lloyd's + RBF vs SVM/RBF simulation for final exam Q14 of the course CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ14Simulation()
    {
      const int EXPERIMETS = 500, TEST_POINTS = 10000;

      var results = Enumerable.Range(0, EXPERIMETS).Select(i =>
        {
          var training = GenerateSinDataSet(100);
          var test = GenerateSinDataSet(TEST_POINTS);

          var rbfResults = RunLloydsRbf(training, test, 9, 1.5);
          var svmResults = RunSvm(training, test);

          return new { 
            failed = !rbfResults.Item1,
            rbfEin = rbfResults.Item2,
            rbfEout = rbfResults.Item3,
            svmEin = svmResults.Item1,
            svmEout = svmResults.Item2,
          };
        }).Where(v => !v.failed).ToArray();
      
      Console.Out.WriteLine("Exam Q14:");
      Console.Out.WriteLine("\tSVM wins: {0}%", (results.Count(v => v.svmEout < v.rbfEout) + 0.0) / results.Length * 100);

      /*      
      Console.Out.WriteLine("\tRuns: {0}", results.Length);
      Console.Out.WriteLine("\tAvg(Ein_rbf): {0}", (results.Sum(v => v.rbfEin) + 0.0) / results.Length);
      Console.Out.WriteLine("\tAvg(Eout_rbf): {0}", (results.Sum(v => v.rbfEout) + 0.0) / results.Length);
      Console.Out.WriteLine("\tAvg(Ein_svm): {0}", (results.Sum(v => v.svmEin) + 0.0) / results.Length);
      Console.Out.WriteLine("\tAvg(Eout_svm): {0}", (results.Sum(v => v.svmEout) + 0.0) / results.Length);*/
    }

    /// <summary>
    /// Lloyd's RBF vs SVM/RBF simulation for final exam Q15 of the course CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ15Simulation()
    {
      const int EXPERIMETS = 500, TEST_POINTS = 10000;

      var results = Enumerable.Range(0, EXPERIMETS).Select(i =>
      {
        var training = GenerateSinDataSet(100);
        var test = GenerateSinDataSet(TEST_POINTS);

        var rbfResults = RunLloydsRbf(training, test, 12, 1.5);
        var svmResults = RunSvm(training, test);

        return new
        {
          failed = !rbfResults.Item1,
          rbfEout = rbfResults.Item3,
          svmEout = svmResults.Item2
        };
      }).Where(v => !v.failed).ToArray();

      Console.Out.WriteLine("Exam Q15:");
      Console.Out.WriteLine("\tSVM wins: {0}%", (results.Count(v => v.svmEout < v.rbfEout) + 0.0) / results.Length * 100);
    }

    /// <summary>
    /// Lloyd's + RBF simulation for final exam Q16 of the course CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ16Simulation()
    {
      const int EXPERIMETS = 500, TEST_POINTS = 10000;

      var results = Enumerable.Range(0, EXPERIMETS).Select(i =>
      {
        var training = GenerateSinDataSet(100);
        var test = GenerateSinDataSet(TEST_POINTS);

        var run1 = RunLloydsRbf(training, test, 9, 1.5);
        var run2 = RunLloydsRbf(training, test, 12, 1.5);

        return new
        {
          failed = !run1.Item1 || !run2.Item1,
          run1Ein = run1.Item2,
          run1Eout = run1.Item3,
          run2Ein = run2.Item2,
          run2Eout = run2.Item3
        };
      }).Where(v => !v.failed).Select(v =>
        {
          if (v.run2Ein < v.run1Ein && v.run2Eout > v.run1Eout) return "a";
          else if (v.run2Ein > v.run1Ein && v.run2Eout < v.run1Eout) return "b";
          else if (v.run2Ein > v.run1Ein && v.run2Eout > v.run1Eout) return "c";
          else if (v.run2Ein < v.run1Ein && v.run2Eout < v.run1Eout) return "d";
          else return "none";
        }).GroupBy(v => v).Select(v => new { option = v.Key, count = v.Count()}).ToArray();

      Console.Out.WriteLine("Exam Q16:");
      foreach(var v in results.Where(u => u.option != "none").OrderByDescending(u => u.count))
        Console.Out.WriteLine("\t({0}):\t{1}%", v.option, v.count * 100.0 / results.Sum(u => u.count));
    }

    /// <summary>
    /// Lloyd's + RBF simulation for final exam Q17 of the course CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ17Simulation()
    {
      const int EXPERIMETS = 500, TEST_POINTS = 10000;

      var results = Enumerable.Range(0, EXPERIMETS).Select(i =>
      {
        var training = GenerateSinDataSet(100);
        var test = GenerateSinDataSet(TEST_POINTS);
        
        var run1 = RunLloydsRbf(training, test, 9, 1.5);
        var run2 = RunLloydsRbf(training, test, 9, 2.0);

        return new
        {
          failed = !run1.Item1 || !run2.Item1,
          run1Ein = run1.Item2,
          run1Eout = run1.Item3,
          run2Ein = run2.Item2,
          run2Eout = run2.Item3
        };
      }).Where(v => !v.failed).Select(v =>
      {
        if (v.run2Ein < v.run1Ein && v.run2Eout > v.run1Eout) return "a";
        else if (v.run2Ein > v.run1Ein && v.run2Eout < v.run1Eout) return "b";
        else if (v.run2Ein > v.run1Ein && v.run2Eout > v.run1Eout) return "c";
        else if (v.run2Ein < v.run1Ein && v.run2Eout < v.run1Eout) return "d";
        else return "none";
      }).GroupBy(v => v).Select(v => new { option = v.Key, count = v.Count() }).ToArray();

      Console.Out.WriteLine("Exam Q17:");
      foreach (var v in results.Where(u => u.option != "none").OrderByDescending(u => u.count))
        Console.Out.WriteLine("\t({0}):\t{1}%", v.option, v.count * 100.0 / results.Sum(u => u.count));
    }

    /// <summary>
    /// Lloyd's + RBF simulation for final exam Q18 of the course CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ18Simulation()
    {
      const int EXPERIMETS = 500, TEST_POINTS = 10000;

      var results = Enumerable.Range(0, EXPERIMETS).Select(i =>
      {
        var training = GenerateSinDataSet(100);
        var test = GenerateSinDataSet(TEST_POINTS);

        var run = RunLloydsRbf(training, test, 9, 1.5);
        return new
        {
          failed = !run.Item1,
          Ein = run.Item2
        };
      }).Where(v => !v.failed).ToArray();

      Console.Out.WriteLine("Exam Q18:");
      Console.Out.WriteLine("\tEin=0:\t{0}%", results.Count(v => v.Ein == 0) * 100.0 / results.Length);
    }

    static Tuple<bool, double, double> RunLloydsRbf(svm_problem training, svm_problem test, int K, double gamma)
    {
      Func<double[], double[], double> Distance = (x, y) => (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]);

      Random rnd = new Random();

      //use LLoyd's to cluster points -----------------------------------------------------------------------------------------------
      //pick K points at random as cluster centers
      var mu = training.x.Select(v => v.Select(u => u.value).ToArray()).OrderBy(v => rnd.NextDouble()).Take(K).ToArray();
      while(true)
      {
        //cluster points
        var clusters = training.x.Select(v => new { x = v, cluster = Enumerable.Range(0, K).OrderBy(cluster =>
          Distance(mu[cluster], v.Select(u => u.value).ToArray())).First() }).OrderBy(v => v.cluster).GroupBy(
          v => v.cluster, v => v.x, (v, u) => u.ToArray()).ToArray();

        //check for empty clusters
        if (clusters.Any(v => v.Length == 0))
          return Tuple.Create(false, 0.0, 0.0);

        //get new mus
        var newMu = clusters.Select(v => new[] { v.Sum(u => u[0].value) / v.Length, v.Sum(u => u[1].value) / v.Length }).ToArray();

        if (mu.Zip(newMu, (v, u) => v[0] == u[0] && v[1] == u[1]).All(v => v))
          break;
        
        mu = newMu;
      }

      //solve to find W
      var Phi = new DenseMatrix(training.l, K + 1);
      var Y = new DenseVector(training.l);
      for (int i = 0; i < training.l; i++)
      {
        Y[i] = training.y[i];

        Phi[i, 0] = 1;
        for (int j = 0; j < K; j++)
          Phi[i, j+1] = Math.Exp(-gamma * Distance(training.x[i].Select(v => v.value).ToArray(), mu[j]));
      }

      var W = Phi.TransposeThisAndMultiply(Phi).Inverse().TransposeAndMultiply(Phi).Multiply(Y);

      Func<double[], double> h = x => W[0] + W.Skip(1).Zip(mu, (w, m) => w * Math.Exp(-gamma * Distance(x, m))).Sum();

      Func<svm_problem, double> E = data => (Enumerable.Range(0, data.l).Count(i => Math.Sign(h(data.x[i].Select(v => v.value).ToArray())) !=
        Math.Sign(data.y[i])) + 0.0) / data.l;

      return Tuple.Create(true, E(training), E(test));
    }

    static Tuple<double, double> RunSvm(svm_problem training, svm_problem test)
    {
      var model = svm.svm_train(training, new svm_parameter()
      {
        svm_type = (int)SvmType.C_SVC,
        kernel_type = (int)KernelType.RBF,
        C = 100000,
        gamma = 1.5,
        eps = 0.001
      });

      Func<svm_problem, double> E = problem => (problem.x.Zip(problem.y, (v, u) => new { x = v, y = u }).Count(v =>
        svm.svm_predict(model, v.x) != v.y) + 0.0) / problem.x.Length;

      return Tuple.Create(E(training), E(test));
    }

    static svm_problem GenerateSinDataSet(int N = 100)
    {
      Random rnd = new Random();
      
      var points = Enumerable.Repeat(0, N).Select(v => new { x1 = rnd.NextDouble() * 2 - 1, x2 = rnd.NextDouble() * 2 - 1 }).ToArray();
      return new svm_problem()
      {
        x = points.Select(v =>
          new svm_node[] { 
              new svm_node() { index = 0, value = v.x1 }, 
              new svm_node() { index = 1, value = v.x2 } }).ToArray(),
        y = points.Select(v => (double)Math.Sign(v.x2 - v.x1 + 0.25 * Math.Sin(Math.PI * v.x1))).ToArray(),
        l = points.Length
      };
    }

    static svm_problem Get1vsAllData(double[][] dataSet, int digit)
    {
      return new svm_problem()
      {
        x = dataSet.Select(v =>
          new svm_node[] { 
              new svm_node() { index = 0, value = v[1] }, 
              new svm_node() { index = 1, value = v[2] } }).ToArray(),
        y = dataSet.Select(v => (v[0] == digit ? 1.0 : -1.0)).ToArray(),
        l = dataSet.Length
      };
    }

    static svm_problem Get1vs1Data(double[][] dataSet, int digitA, int digitB)
    {
      var subset = dataSet.Where(v => v[0] == digitA || v[0] == digitB);
      return new svm_problem()
      {
        x = subset.Select(v =>
          new svm_node[] { 
                  new svm_node() { index = 0, value = v[1] }, 
                  new svm_node() { index = 1, value = v[2] } }).ToArray(),
        y = subset.Select(v => (v[0] == digitA ? 1.0 : -1.0)).ToArray(),
        l = subset.Count()
      };
    }

    static private Tuple<double, double> RunRegularizedRegression(
      svm_problem trainingData, 
      svm_problem testData, 
      Func<double, double, double[]> zTransform,
      double lambda = 1.0)
    {
      int zLength = zTransform(0, 0).Length;

      int N = trainingData.l;
      var Z = new DenseMatrix(N, zLength);
      var Y = new DenseVector(N);
      for (int j = 0; j < N; j++)
      {
        Y[j] = trainingData.y[j];

        var zj = zTransform(trainingData.x[j][0].value, trainingData.x[j][1].value);
        for (int k = 0; k < zLength; k++)
          Z[j, k] = zj[k];
      }

      var W = Z.TransposeThisAndMultiply(Z).Add(DenseMatrix.Identity(zLength).Multiply(lambda)).Inverse().TransposeAndMultiply(Z).Multiply(Y);

      Func<double, double, double> h = (x1, x2) => W.DotProduct(DenseVector.OfEnumerable(zTransform(x1, x2))) >= 0 ? 1.0 : -1.0;
      Func<svm_problem, double> E = data => (Enumerable.Range(0, data.l).Count(i => Math.Sign(h(data.x[i][0].value, data.x[i][1].value)) !=
        Math.Sign(data.y[i])) + 0.0) / data.l;

      return Tuple.Create(E(trainingData), E(testData));
    }

  }
}
