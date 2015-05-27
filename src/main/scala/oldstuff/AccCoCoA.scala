//package optimizers.distributed
//
//import breeze.linalg.{DenseVector, Vector}
//import models.{Regularizer, RealFunction, Loss}
//import optimizers.DistributedOptimizer
//import org.apache.spark.SparkContext
//import org.apache.spark.rdd.RDD
//import utils.OptUtils
//import utils.OptUtils._
//import vectors.{LazySumVector, LabelledPoint}
//import vectors.VectorOps._
//
//class AccCoCoA[-LossType<:Loss[RealFunction,RealFunction]]
//  (sc: SparkContext, distopt: DistributedOptimizer[LossType], seed: Int)
//    extends DistributedOptimizer [LossType] {
//
//  override def optimize (
//    loss: LossType, regularizer: Regularizer, n: Long,
//    data: RDD[Array[LabelledPoint]],
//    alphaInit: RDD[DenseVector[Double]],
//    vInit: DenseVector[Double],
//    epsilon: Double) : (RDD[DenseVector[Double]], DenseVector[Double]) = {
//
//    val lambda = regularizer.lambda
//
//    val R = math.sqrt(data.flatMap(_.map(pt => l2norm(pt.features))).max)
//    val gamma = 4.0
//    val kappa = R*R/(gamma*n) - lambda
//
////    val kappa = 0.5
//
//    if (R*R/(gamma*n) <= 10*n) {
//      println("Solving using standar CoCoA")
//      return distopt.optimize(loss,regularizer,n,data,alphaInit,vInit,epsilon)
//    }
//
//    val mu = lambda / 2
//    val rho = mu + kappa
//
//    val eta = math.sqrt(mu/rho)
//
//    val bbeta = (1-eta)/(1+eta)
//
//    var alpha = alphaInit
//
//    val d = data.first()(0).features.size
//
//    var v : DenseVector[Double] = DenseVector.zeros[Double](d)
//    var w : Vector[Double] = DenseVector.zeros[Double](d)
//    var y = DenseVector.zeros[Double](d)
//
//    var xi = (1 + 1/(eta*eta)) *
//      (computePrimalObjective(data, loss, regularizer, n, regularizer.dualGradient(v))
//        - computeDualObjective(data, loss, regularizer, n, v, alpha))
//
//    var accreg = regularizer
//
//    for (t <- 2 to 100) {
//
//      val eps = xi * eta / (2*(1 + 1/(eta*eta)))
//
//      accreg = new AccRegularizer(regularizer, kappa, y)
//
//      y = (w * (-bbeta)).toDenseVector
//
//      val asdf = distopt.optimize(loss, accreg, n, data, alpha, v, eps)
//
//      v = asdf._2
//      w = accreg.dualGradient(v)
//      alpha = asdf._1
//
//      y = (w * (1 + bbeta) + y).toDenseVector
//
//      xi *= 1 - eta/2
//
//      OptUtils.printSummaryStatsPrimalDual("AccCoCoA", data, loss,
//        regularizer, n,
//        (data zip alpha).flatMap(p=> p._1 zip p._2.data)
//          .map { case(LabelledPoint(y,x), a) => x * (a/(lambda*n)) }.reduce(_+_).toDenseVector,
//        alpha)
//    }
//
//    (alpha, v)
//  }
//
//  class AccRegularizer(reg: Regularizer, kappa: Double, y: DenseVector[Double]) extends Regularizer {
//
//    def lambda = reg.lambda
//
//    def primal(w: Vector[Double]) = reg.primal(w) - kappa*(w dot y)
//
//    //  def dualGradient(w: Vector[Double]) = reg.dualGradient(w + (y * kappa))
//    def dualGradient(w: Vector[Double]) = w match {
//      case (w: DenseVector[Double]) => reg.dualGradient(new LazySumVector(w, y, kappa))
//    }
//  }
//}
