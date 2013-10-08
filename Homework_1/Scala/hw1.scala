package lfd

import scala.util.Random

object hw1 {

  type Point = (Double, Double)

  // next random point
  def nextR = (1-2*Random.nextDouble, 1-2*Random.nextDouble) // "1 -" adjusts the scale to -1 to 1

  // random line in the range
  def lineR = (nextR, nextR)

  // cross product function to find which side of a-b does c lie. +ve or -ve
  def side(a:Point, b:Point)( c:Point) = 
    ((b._1 - a._1)*(c._2 - a._2) - (b._2 - a._2)*(c._1 - a._1)) >= 0;

  // run PLA for N points
  def runPLA(N:Int) : (Long, Double) = {
    val lines = lineR

    // f and g function. f calculates actual side based on line
    // g calculates side based w
    val f = side(lineR._1, lineR._2) _
    val g = (w: (Double, Double, Double), p: (Double, Double, Double)) => {
      w._1*p._1 + w._2*p._2 + w._3*p._3 >= 0
    }

    // the N random points and the sides they lie on calculated by f
    val points = (for (i <- 1 to N) yield { nextR }).to[Vector]
    val fSides = points.map { f }

    // init
    var (w0, w1, w2) = (0.0, 0.0, 0.0)

    // TODO - not pretty, not pretty at all. procedural construct and not functional!!!
    var iter = 1
    var end = 50000
    var converged = 0

    while (iter <= end){
      // calcualate the side which points lie using g
      val gSides = points.map { p => g( (w0, w1, w2), (1, p._1, p._2)) }
      // find where f and g disagree, and also the index of the point where they disagree
      val notEq = fSides.zip(gSides).zipWithIndex.filter { case ((f, g), i) => f != g }

      if (notEq.size > 0) {
        val pickOne = notEq(0)
        val index = pickOne._2
        val sign = if (pickOne._1._1 == false ) -1 else +1  // if f(x) which is y is false then -1 else +1
        w0 = w0 + sign*1
        w1 = w1 + sign*points(index)._1
        w2 = w2 + sign*points(index)._2
      } else {
        // TODO - not pretty, not pretty at all. procedural construct and not functional!!!
        converged = iter
        iter = end
      }
      iter = iter + 1
    }

    // TODO - not pretty, not pretty at all. procedural construct and not functional!!!
    if (converged == 0)
      (0, 0)
    else {
      // a new g function which uses the converged w's to find the side
      val g1 = (p: Point) => {
        g((w0, w1, w2), (1, p._1, p._2))
      }
      // where converged, and P[F != G] 
      (converged, pFNotG(f, g1))
    }
  }

  // evaluate P[F != G] for a sample
  def pFNotG(F: (Point => Boolean), G:(Point => Boolean)): Double = {
    val SAMPLE = 10000
    val fNotG = (1 to SAMPLE).foldLeft( 0.0 ) {
      (fNotGAccum, i) => // ignore i
         val p = nextR
         (F(p) != G(p)) match x{
           case true  => fNotGAccum + 1
           case false => fNotGAccum
         }
    }
    fNotG / SAMPLE
  }

  // call with N=10 or N=100 for simulating PLA
  def simulatePLA(N:Int) = {
    val SIM = 10000
    val (converged, fNotG) = (1 to SIM).foldLeft( (0.0, 0.0) ) {
      case ((converged, fNotG), i) => // ignore i
        val (_c, _fNotG) = runPLA(N)
        (converged + _c, fNotG + _fNotG)
    }
    println("Run for N = " + N)
    println("Average of convergence = " + converged / SIM)
    println("Average of f not g = " + fNotG / SIM)
  }
}
