/*
 * This file contains C#/.NET 4.5 implementations for 8th week homework of the CS1156x "Learning From Data" at eDX
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
using System.Net;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using libsvm;

namespace StochasticTinker.edX.CS1156x.HW8 
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

  class HW8
  {
    static double[][] trainingData = Regex.Split(new WebClient().DownloadString("http://www.amlbook.com/data/zip/features.train"), "\r\n|\r|\n").Where(
        line => !string.IsNullOrWhiteSpace(line)).Select(
        line => line.Trim().Replace("   ", ",").Replace("  ", ",").Split(',').Select(v => Convert.ToDouble(v)).ToArray()).ToArray();

    static double[][] testData = Regex.Split(new WebClient().DownloadString("http://www.amlbook.com/data/zip/features.test"), "\r\n|\r|\n").Where(
        line => !string.IsNullOrWhiteSpace(line)).Select(
        line => line.Trim().Replace("   ", ",").Replace("  ", ",").Split(',').Select(v => Convert.ToDouble(v)).ToArray()).ToArray();

    static HW8()
    {
      //Supress svmlib console output
      svm.svm_set_print_string_function(new SvmSilencer());
    }

    static void Main(string[] args)
    {
      RunQ2_4Simulation();
      RunQ5Simulation();
      RunQ6Simulation();
      RunQ7_8Simulation();
      RunQ9_10Simulation();
    }

    /// <summary>
    /// Polynomial kernel SVM simulation for homework Q2 - Q4 of the 
    ///  8th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ2_4Simulation()
    {
      var results = Enumerable.Range(0, 10).Select(d =>
        {
          var v = Run1VsAll(d);
          return new { digit = d, eIn = v.Item1, svCount = v.Item2 };
        }).ToArray();

      var q2Result = results.Where(v => v.digit % 2 == 0).OrderBy(v => v.eIn).Last();

      Console.Out.WriteLine("HW8 Q2:");
      Console.Out.WriteLine("\tMax(Ein) = {0}", q2Result.eIn);
      Console.Out.WriteLine("\t# of SVs = {0}", q2Result.svCount);
      Console.Out.WriteLine("\tAchieved on {0} vs all", q2Result.digit);

      var q3Result = results.Where(v => v.digit % 2 == 1).OrderBy(v => v.eIn).First();

      Console.Out.WriteLine("HW8 Q3:");
      Console.Out.WriteLine("\tMin(Ein) = {0}", q3Result.eIn);
      Console.Out.WriteLine("\t# of SVs = {0}", q3Result.svCount);
      Console.Out.WriteLine("\tAchieved on {0} vs all", q3Result.digit);

      Console.Out.WriteLine("HW8 Q4:");
      Console.Out.WriteLine("\t# of SVs difference = {0}", Math.Abs(q3Result.svCount - q2Result.svCount));
    }

    /// <summary>
    /// Polynomial kernel SVM simulation for homework Q5 of the 
    ///  8th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ5Simulation()
    {
      Console.Out.WriteLine("HW8 Q5:");
      foreach (double C in new [] {0.001, 0.01, 0.1, 1})
      {
        var result = Run1Vs1(1, 5, C, 2);
        Console.Out.WriteLine("\tC = {0}", C);
        Console.Out.WriteLine("\tEin = {0}", result.Item1);
        Console.Out.WriteLine("\tEout = {0}", result.Item2);
        Console.Out.WriteLine("\t# of SVs = {0}", result.Item3);
      }
    }

    /// <summary>
    /// Polynomial kernel SVM simulation for homework Q6 of the 
    ///  8th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ6Simulation()
    {
      Console.Out.WriteLine("HW8 Q6:");
      foreach (var C in new[] { 0.0001, 0.001, 0.01, 1 })
      {
        Console.Out.WriteLine("\tC = {0} -------------------", C);
        foreach (var Q in new[] { 2, 5 })
        {
          var result = Run1Vs1(1, 5, C, Q);
          Console.Out.WriteLine("\tQ = {0}", Q);
          Console.Out.WriteLine("\tEin = {0}", result.Item1);
          Console.Out.WriteLine("\tEout = {0}", result.Item2);
          Console.Out.WriteLine("\t# of SVs = {0}", result.Item3);
          Console.Out.WriteLine();
        }
        Console.Out.WriteLine();
      }
    }

    /// <summary>
    /// Cross-validation of C parameter in polynomial kernel SVM for homework Q7-8 of the 
    ///  8th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ7_8Simulation()
    {
      var result = Enumerable.Repeat(1, 100).Select(v =>
        {
          long seed = DateTime.Now.Ticks;
          return new[] { 0.0001, 0.001, 0.01, 0.1, 1 }.Select(C =>
            new
            {
              C = C,
              Ecv = Enumerable.Repeat(1, 2).Average(i => CrossValidate(seed, C))
            }).ToArray().OrderBy(u => u.Ecv).ThenBy(u => u.C).First();
         }).GroupBy(v => v.C).Select(group => 
           new { C = group.Key, Count = group.Count(), Ecv = group.Average(v => v.Ecv) }).ToArray();

      Console.Out.WriteLine("HW8 Q7:");
      Console.Out.WriteLine("\tC = {0}", result.OrderBy(v => v.Count).Last().C);
      Console.Out.WriteLine("\t# of wins = {0}", result.OrderBy(v => v.Count).Last().Count);

      Console.Out.WriteLine("HW8 Q8:");
      Console.Out.WriteLine("\tAcg(Ecv) = {0}", result.OrderBy(v => v.Count).Last().Ecv);
    }

    /// <summary>
    /// RBF kernel SVM sumulation for homework Q9-10 of the 
    ///  8th week of the CS1156x "Learning From Data" at eDX
    /// </summary>
    static void RunQ9_10Simulation()
    {
      var result = new[] { 0.01, 1, 100, 10000, 1000000 }.Select(C =>
        {
          var v = RunRbf1Vs1(1, 5, C);

          return new
          {
            C = C,
            Ein = v.Item1,
            Eout = v.Item2,
            svCount = v.Item3
          };
        }).ToArray();

      Console.Out.WriteLine("HW8 Q9:");
      Console.Out.WriteLine("\tMin(Ein) = {0}", result.OrderBy(v => v.Ein).First().Ein);
      Console.Out.WriteLine("\tC = {0}", result.OrderBy(v => v.Ein).First().C);

      Console.Out.WriteLine("HW8 Q10:");
      Console.Out.WriteLine("\tMin(Eout) = {0}", result.OrderBy(v => v.Eout).First().Eout);
      Console.Out.WriteLine("\tC = {0}", result.OrderBy(v => v.Eout).First().C);
    }

    static double CrossValidate(long randomSeed, double C)
    {
      var training = Create1vs1Problem(trainingData, 1, 5);

      var config = new svm_parameter()
      {
        svm_type = (int)SvmType.C_SVC,
        kernel_type = (int)KernelType.POLY,
        C = C,
        degree = 2,
        coef0 = 1,
        gamma = 1,
        eps = 0.001
      };

      double[] result = new double[training.l];
      svm.rand.setSeed(randomSeed);
      svm.svm_cross_validation(training, config, 10, result);
      return (result.Zip(training.y, (v, u) => Math.Sign(v) != Math.Sign(u) ? 1 : 0).Sum() + 0.0) / result.Length;
    }

    /// <summary>
    ///   Runs polynomial-kernel SVM C-classifier on a specified digit vs all other digits 
    /// </summary>
    /// <param name="digit"></param>
    /// <returns>
    ///   Ein, # of support vectors
    /// </returns>
    static Tuple<double, int> Run1VsAll(int digit)
    {
      var prob = new svm_problem()
      {
        x = trainingData.Select(v =>
          new svm_node[] { 
              new svm_node() { index = 0, value = v[1] }, 
              new svm_node() { index = 1, value = v[2] } }).ToArray(),
        y = trainingData.Select(v => (v[0] == digit ? 1.0 : -1.0)).ToArray(),
        l = trainingData.Length
      };

      var model = svm.svm_train(prob, new svm_parameter()
      {
        svm_type = (int)SvmType.C_SVC,
        kernel_type = (int)KernelType.POLY,
        C = 0.01,
        degree = 2,
        coef0 = 1,
        gamma = 1,
        eps = 0.001
      });

      return Tuple.Create((prob.x.Zip(prob.y, (v, u) => new { x = v, y = u }).Count(v =>
        Math.Sign(svm.svm_predict(model, v.x)) != Math.Sign(v.y)) + 0.0) / prob.x.Length,
        model.l);
    }

    /// <summary>
    ///   Runs polynomial-kernel SVM C-classifier on a specified digitA vs digitB
    /// </summary>
    /// <param name="digit"></param>
    /// <returns>
    ///   Ein, Eout, # of support vectors
    /// </returns>
    static Tuple<double, double, int> Run1Vs1(int digitA, int digitB, double C, int Q)
    {
      var training = Create1vs1Problem(trainingData, digitA, digitB);

      var model = svm.svm_train(training, new svm_parameter()
      {
        svm_type = (int)SvmType.C_SVC,
        kernel_type = (int)KernelType.POLY,
        C = C,
        degree = Q,
        coef0 = 1,
        gamma = 1,
        eps = 0.001,
        shrinking = 1   //does not complete in reasonable time for C=1, Q=5 without this parameter!
      });

      var test = Create1vs1Problem(testData, digitA, digitB);

      Func<svm_problem, double> E = p => (p.x.Zip(p.y, (v, u) => new { x = v, y = u }).Count(v =>
        Math.Sign(svm.svm_predict(model, v.x)) != Math.Sign(v.y)) + 0.0) / p.x.Length;

      return Tuple.Create(E(training), E(test), model.l);
    }

    /// <summary>
    ///   Runs RBF-kernel SVM C-classifier on a specified digitA vs digitB
    /// </summary>
    /// <param name="digit"></param>
    /// <returns>
    ///   Ein, Eout, # of support vectors
    /// </returns>
    static Tuple<double, double, int> RunRbf1Vs1(int digitA, int digitB, double C)
    {
      var training = Create1vs1Problem(trainingData, digitA, digitB);

      var model = svm.svm_train(training, new svm_parameter()
      {
        svm_type = (int)SvmType.C_SVC,
        kernel_type = (int)KernelType.RBF,
        C = C,
        coef0 = 1,
        gamma = 1,
        eps = 0.001,
        shrinking = 1
      });

      var test = Create1vs1Problem(testData, digitA, digitB);

      Func<svm_problem, double> E = p => (p.x.Zip(p.y, (v, u) => new { x = v, y = u }).Count(v =>
        Math.Sign(svm.svm_predict(model, v.x)) != Math.Sign(v.y)) + 0.0) / p.x.Length;

      return Tuple.Create(E(training), E(test), model.l);
    }

    static svm_problem Create1vs1Problem(double[][] data, int digitA, int digitB)
    {
      var subset = data.Where(v => v[0] == digitA || v[0] == digitB);
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
  }
}
