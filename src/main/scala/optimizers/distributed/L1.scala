package optimizers.distributed

import breeze.linalg.{SparseVector, DenseVector, Vector}
import models._
import optimizers.SingleCoordinateOptimizer
import optimizers.coordinate.{BrentMethodOptimizerWithFirstDerivative, BrentMethodOptimizer}
import optimizers.local.SDCAOptimizer
import org.apache.commons.math.analysis.UnivariateRealFunction
import org.apache.commons.math.analysis.solvers.BrentSolver
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import test.ConjugateTest
import utils.OptUtils
import vectors.{LazyMappedVector, LazySumVector, LazyScaledVector, LabelledPoint}
import vectors.VectorOps._

import scala.Double._
import scala.math._

object L1 {

  def optimize(sc: SparkContext, Xt: RDD[LabelledPoint], lam: Double, numPasses: Int, numRounds: Int, numSplits: Int) = {

    // number of passes of the local solver
//    val numPasses = 1
    // number of cocoa rounds
//    val numRounds = 100
//    val numSplits = 1

    val (d,n) = (Xt.count().toInt, Xt.first().features.size)
//println(d,n)
    val b = new DenseVector[Double](Xt.map(_.label).toLocalIterator.toArray)
    val A = transpose(Xt, d, n, lam, sc).coalesce(numSplits)

    val partData = A.mapPartitions(x=>Iterator(x.toArray), preservesPartitioning = true)
    val partAlphas = A.map(_ => 0.0)
      .mapPartitions(x=>Iterator(new DenseVector(x.toArray)), preservesPartitioning = true)

    val v = DenseVector.zeros[Double](d)

    val lambda = 1.0/(lam*d*n)
    val regularizer = new L2RegularizerMinusB(b)
    val loss = new LModifiedL1Loss(regularizer.dual(v)/(d*lam))

    val model = new Model (d, lambda, loss, regularizer)

    val scOptimizer = new BrentMethodOptimizerWithFirstDerivative(1000)

    val localSolver = new SDCAOptimizer(scOptimizer, numPasses)

    val cocoa = new CoCoA(sc, localSolver)

    cocoa.optimize(model, partData, partAlphas, v)
  }

  def transpose(At: RDD[LabelledPoint], d: Int, n: Int, lambda: Double, sc: SparkContext): RDD[LabelledPoint] = {
    (At.map(_.features.asInstanceOf[SparseVector[Double]]) ++
      sc.parallelize(Array(new SparseVector[Double]((0 until n).toArray, Array.fill(n)(0), d))))
        .zipWithIndex()
          .flatMap{case (p,i) => (p.index zip p.data).map{case (f,v) => (f,(i.toInt,v))}}
            .groupBy(_._1).sortBy(_._1).map {case (i, p) => (i, p.map(_._2))}.map(_._2.toArray.sorted)
              .map(dd => new LabelledPoint(0.0, new SparseVector[Double](
                dd.map(_._1).toArray, dd.map(_._2).toArray, dd.size-1, d) / (lambda*d)))
  }
}

class LModifiedL1Loss(val B: Double) extends GenericLoss[RealFunction,DifferentiableRealFunction] (
  (y:Double) => (
    (x:Double) => {
      if (abs(x) <= 1.0)
        0.0
      else
        B*(abs(x) - 1)
    },
    (NegativeInfinity, PositiveInfinity)
    ),

  (y:Double) => (
    (a:Double) => {
      if (abs(a) <= B)
        abs(a)
      else
        PositiveInfinity
    },
    (a:Double) => {
      assert(abs(a) <= B)
      signum(a)
    },
    (-B, B)
  )
)

class L2RegularizerMinusB(b: Vector[Double]) extends Regularizer {

  def primal(w: Vector[Double]) = l2norm(w) / 2 + (w dot b)
  def dualGradient(v: Vector[Double]) = new LazySumVector(
    v.asInstanceOf[DenseVector[Double]], b.asInstanceOf[DenseVector[Double]], -1.0)
}

class LogisticRegularizerMinusB(b: Vector[Double]) extends Regularizer {

  def primal(w: Vector[Double]) = w.toArray.zipWithIndex.map { case(a,i) =>
    if (a == 0 || b(i)*a == -1) 0.0 else (1+a*b(i))*log(1+a*b(i)) - a*b(i)*log(-a*b(i))
  }.sum

  def dualGradient(v: Vector[Double]) = new LazyMappedVector(v, {
    case(i,x) => -b(i) / (1 + exp(b(i)*x))
  })

  override def dual(v: Vector[Double]) = {
    val w = dualGradient(v)
    (w dot v) - primal(w)
  }
}

//class SoftThresholdingSolver extends SingleCoordinateOptimizer[QModifiedL1Loss] {
//  def optimize(loss: QModifiedL1Loss, regularizer: Regularizer, n: Long,
//               pt: LabelledPoint, alpha: Double, v: Vector[Double], epsilon: Double = 0.0) = {
//
//    val lambda = regularizer.lambda
//    val w = regularizer.dualGradient(v)
//
//    val x = pt.features
//
//    val (xx, xw) = (x dot x, x dot w)
//
//    val sigma = lambda/loss.kappa
//    val r = alpha*xx - xw
//    val deltaAlpha = if (xx == 0) {
//      0.0
//    } else if (r - sigma > 0.0) {
//      (r - sigma) / xx - alpha
//    } else if (r + sigma < 0.0) {
//      (r + sigma) / xx - alpha
//    } else {
//      - alpha
//    }
//
//    assert(abs(alpha + deltaAlpha) <= loss.B/(n*loss.kappa))
//
//    //      val deltaAlpha = if (xx > 0.0) {
//    //        val r = (xx*alpha - xw)/xx
//    //        signum(r) * max(abs(r) - loss.kappa/(lambda*xx), 0.0) - alpha
//    //      } else {
//    //        0.0
//    //      }
//
//    (deltaAlpha, new LazyScaledVector(x, deltaAlpha / (lambda * n)))
//  }
//}
//
//class QModifiedL1Loss(val B: Double) extends GenericLoss[RealFunction,DifferentiableRealFunction] (
//  (y:Double) => (
//    (x:Double) => {
//      if (abs(x) <= 1.0)
//        0.0
//      else
//        1.0/4 * pow(abs(x),2) + (2*B - 1)/2 * abs(x) - (4*B - 1)/4
//    },
//    (NegativeInfinity, PositiveInfinity)
//    ),
//
//  (y:Double) => (
//    (a:Double) => {
//      if (abs(a) <= B)
//        abs(a)
//      else
//        abs(a) + pow(abs(a) - B, 2)
//    },
//    (a:Double) => {
//      assert(abs(a) <= B)
//      signum(a)
//    },
//    (-B, B)
//    )
//)