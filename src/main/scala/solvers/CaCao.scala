package distopt.solvers

import java.security.SecureRandom
import java.util.Calendar
import distopt.utils
import distopt.utils.VectorOps._
import distopt.utils._
import localsolvers.LocalSolverTrait
import org.apache.commons.math.analysis.UnivariateRealFunction
import org.apache.commons.math.analysis.DifferentiableUnivariateRealFunction
import org.apache.commons.math.optimization.GoalType
import org.apache.commons.math.optimization.univariate.BrentOptimizer
import org.apache.commons.math.analysis.solvers.{BrentSolver,NewtonSolver,MullerSolver}
import org.apache.commons.math3.analysis.{DifferentiableUnivariateFunction, UnivariateFunction}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors, DenseVector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.commons.math3.analysis.solvers.NewtonRaphsonSolver
import models.RealFunction

object CaCaO {

  /**
   * CaCao - Communication-efficient distributed SGD.
   * 
   * @param sc Spark context
   * @param data RDD of all data examples
   * @param wInit initial weight vector (has to be zero)
   * @param numRounds number of outer iterations T in the paper
   * @param beta scaling parameter. beta=1 gives averaging, beta=K=data.partitions.size gives (aggressive) adding
   * @param chkptIter checkpointing the resulting RDDs from time to time, to ensure persistence and shorter dependencies
   * @param testData ...
   * @param debugIter ...
   * @param seed ...
   * @return
   */
  def runCaCaO(
    sc: SparkContext, 
    data: RDD[Array[LabeledPoint]],
    wInit: Vector,
    numRounds: Int,
    beta: Double, 
    chkptIter: Int,
    n: Int,
    testData: RDD[LabeledPoint],
    debugIter: Int,
    lambda: Double,
    primalLoss: RealFunction,
    sgdIterations: Int,
    seed: Int) : Vector = {
    
    val parts = data.partitions.size

    println("\nRunning CaCaO on "+n+" data examples, distributed over "+parts+" workers")

    val scaling = beta / parts

    var w = wInit

    var wLocal = data.mapPartitions(x => Iterator(Vectors.zeros(w.size)), preservesPartitioning=true)

    for(t <- 1 to numRounds){

      val updates = (wLocal zip data).mapPartitions(
        partitionUpdate(_,w,lambda,n,sgdIterations,seed+t),preservesPartitioning=true).persist()

      val primalUpdates = updates.reduce(plus)

      w = plus(times(primalUpdates,scaling),w)

      wLocal = (wLocal zip updates).map{ case (wK,delta) => plus(wK, times(delta,scaling)) }

      println(s"Iteration: $t")
      println(w(0))
//      OptUtils.printSummaryStats("CaCaO",data,w,lambda,testData,primalLoss)
    }

    w
  }

  /**
   * Performs one round of local updates using a given local dual algorithm, 
   * here locaSDCA. Will perform localIters many updates per worker.
   * @param zipData ...
   * @param w ...
   * @param seed ...
   * @return
   */
  private def partitionUpdate(
    zipData: Iterator[(Vector, Array[LabeledPoint])],
    w: Vector,
    lambda: Double,
    n: Int,
    sgdIterations: Int,
    seed: Int): Iterator[Vector] = {

    val zipPair = zipData.next()

    val localData = zipPair._2
    val oldWk = zipPair._1
    val wBar = plus(w, times(oldWk, -1))

    val newWk = solve(localData, wBar, oldWk, lambda, n, sgdIterations, seed)

    Iterator(plus(newWk, times(oldWk, -1)))
  }

  private def solveSPCA(data: Array[LabeledPoint], wBar: Vector, wKold: Vector, lambda: Double, n: Int, a: Int, seed: Int): Vector = {

    val wK = wKold.copy

    val rand = new SecureRandom
    rand.setSeed(seed)

    val nonZero = (0 until wBar.size).map(i => (0 until data.size).filter(data(_).features(i) != 0.0).toArray).toArray

    var norm = dot(wK,wK)

    val dotProd = data.map(pt => dot(pt.features,wK) + dot(pt.features,wBar)).toArray

//    var avgLoss = (data zip dotProd).map{case (pt,prod) => logLoss(prod,pt.label)/n}.sum

    for (it <- 1 to wK.size) {
      val j = rand.nextInt(wK.size)

      val oldwj = wK.asInstanceOf[DenseVector].values(j)

//      val func = new UnivariateRealFunction {
//        def value(wj: Double) = {
//
//          var loss = avgLoss
//
//          for (i <- nonZero(j)) {
//            loss += logLoss(innProd(i) + (wj - oldwj) * data(i).features(j), data(i).label) / n
//              - logLoss(innProd(i), data(i).label) / n
//          }
//
//          loss + lambda * (norm + wj*wj - oldwj*oldwj) / 2
//        }
//      }

      val der = new UnivariateRealFunction {
        def value(wj: Double) = {

          var res = lambda*wj

          for (i <- nonZero(j)) {
            val xj = data(i).features(j)
            val y = data(i).label
            val dot = dotProd(i) + (wj - oldwj) * xj

            res += (-(xj*y) / (1 + math.exp(y*dot))) / n
          }

          res
        }
      }

      val secondderiv = new UnivariateRealFunction {
        def value(wj: Double) = {

          var res = lambda

          for (i <- nonZero(j)) {
            val xj = data(i).features(j)
            val y = data(i).label
            val dot = dotProd(i) + (wj - oldwj) * xj

            res += (math.pow(xj*y,2) * math.exp(y*dot) / math.pow(1 + math.exp(y*dot), 2)) / n
          }

          res
        }
      }

      val function = new DifferentiableUnivariateRealFunction {
        override def derivative = secondderiv
        override def value(a: Double): Double = der.value(a)
      }

//      val brent = new BrentOptimizer
//      brent.setMaximalIterationCount(100)
//      brent.setRelativeAccuracy(0)
//      brent.setAbsoluteAccuracy(2*Math.ulp(1d))
//      val wj = brent.optimize(func, GoalType.MINIMIZE, -10, 10, oldwj)

//      val brent = new BrentSolver
//      brent.setMaximalIterationCount(1000)
//      brent.setRelativeAccuracy(1.0E-4D)
////      brent.setAbsoluteAccuracy(1000*Math.ulp(1d))
//      val wj = brent.solve(function, -10, 10, oldwj)

      val newton = new NewtonSolver
      newton.setMaximalIterationCount(1)
      newton.setAbsoluteAccuracy(Double.MaxValue)
      val wj = newton.solve(function, -10, 10, oldwj)

      if (wj == -10 || wj == 10) throw new RuntimeException("no")

      for (i <- nonZero(j)) {
//        avgLoss -= logLoss(innProd(i), data(i).label) / n
        dotProd(i) += (wj - oldwj)*data(i).features(j)
//        avgLoss += logLoss(innProd(i), data(i).label) / n
      }

      norm += wj*wj - oldwj*oldwj

      wK.asInstanceOf[DenseVector].values(j) = wj
    }
    wK
  }

  private def svmLoss(pt: LabeledPoint, wK: Vector, wBar: Vector) = {
    val (x,y) = (pt.features, pt.label)
    math.max(0,1-dot(plus(wK,wBar),x)*y)
  }

  private def svmLossGradient(pt: LabeledPoint, wK: Vector, wBar: Vector) = {
    val (x,y) = (pt.features, pt.label)

    if (1 - (dot(x,wK) + dot(x,wBar)) * y >= 0)
      times(x,-y)
    else
      Vectors.sparse(wK.size, Array[Int](), Array[Double]())
  }

  private def logLoss(x: Double, y: Double) : Double = math.log(1 + math.exp(-x*y))

  private def logLossGradient(pt: LabeledPoint, wK: Vector, wBar: Vector, s: Double) = {
    val (x,y) = (pt.features, pt.label)

    val exp = math.exp((dot(x,wK)*s + dot(x,wBar))*y)

    times(x,-y/(1 + exp))
  }

  private def solve(data: Array[LabeledPoint], wBar: Vector, wKold: Vector, lambda: Double,
                    n: Int, numIter: Int, seed: Int): Vector = {

    var wK = wKold.copy
    val nK = data.size

    val rand = new SecureRandom
    rand.setSeed(seed)

    var s = 1.0
    for (i <- 1 to numIter) {
      val idx = rand.nextInt(data.size)
      val pt = data(idx)

      val eta = 1.0/math.pow(i,1.0)

//      val lossGrad = times(logLossGradient(pt, wK, wBar, s),-eta*nK.toDouble/n)
//      plusEqual(wK.asInstanceOf[DenseVector], lossGrad, 1/(1 - eta*lambda)/s)
//      s *= (1 - eta*lambda)

//      wK = plus(wK, times(plus(times(logLossGradient(pt, wK, wBar,1.0),nK.toDouble/n),times(wK,lambda)), -eta))

      wK = plus(wK, times(plus(times(logLossGradient(pt, wK, wBar,1.0),nK.toDouble/n),times(
//        2wK
          plus(times(wK, 2*math.log(1 + dot(wK,wK))), times(wK, 2))
        ,0.5*lambda)), -eta))
    }
    println(wK(0))
    times(wK,s)
  }
}
