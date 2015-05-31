package optimizers.distributed

import breeze.linalg.DenseVector
import models._
import optimizers.coordinate.{BrentMethodOptimizer, BrentMethodOptimizerWithFirstDerivative, RidgeOptimizer, SVMOptimizer}
import optimizers.local.SDCAOptimizer
import optimizers.{SingleCoordinateOptimizer, LocalOptimizer, DistributedOptimizer}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import utils.{DualityGapConvergenceChecker, ConvergenceChecker}
import vectors.LabelledPoint

class CoCoA[-LossType<:Loss[_,_]] (
  localSolver: LocalOptimizer[LossType],
  beta: Double = 1.0,
  convergenceChecker: ConvergenceChecker[LossType] = new DualityGapConvergenceChecker)
  (implicit @transient sc: SparkContext)
  extends DistributedOptimizer[LossType] {

  def optimize (
    model: Model[LossType],
    dataRDD: RDD[LabelledPoint],
    alphaInitRDD: RDD[Double],
    vInit: DenseVector[Double]) = {

    val data = dataRDD.mapPartitions(x=>Iterator(x.toArray), preservesPartitioning = true)
    val alphaInit = alphaInitRDD.map(_ => 0.0)
      .mapPartitions(x=>Iterator(new DenseVector(x.toArray)), preservesPartitioning = true)

    data.cache()
    val parts = data.count()
    val scaling = beta / parts

    var alpha = alphaInit
    alpha.cache()
    val v = vInit

    var t = 1

    while (!convergenceChecker.hasConverged(model, data, alpha, v, t)) {

      val vv = sc.broadcast(v)

      val updates = (alpha zip data).mapPartitions(
        CoCoA.partitionUpdate(model, localSolver, _, vv),preservesPartitioning=true).cache()

      alpha = (alpha zip updates.map(_._2)).map ({
        case (alphaOld, deltaAlpha) => alphaOld + deltaAlpha * scaling }).cache()

      v += updates.map(_._1).reduce(_+_) * scaling

      t += 1
    }

    (alpha.flatMap(_.data), v)
  }

  def optimize(model: Model[LossType], dataRDD: RDD[LabelledPoint]): (RDD[Double], DenseVector[Double]) = {
    val numFeatures = dataRDD.first().features.size
    val alpha = dataRDD.map(_=>0.0)
    val v = DenseVector.zeros[Double](numFeatures)
    optimize(model, dataRDD, alpha, v)
  }
}

object CoCoA {
  def partitionUpdate [LossType<:Loss[_,_]] (
    model: Model[LossType],
    localSolver: LocalOptimizer[LossType],
    zipData: Iterator[(DenseVector[Double], Array[LabelledPoint])],
    vInit: Broadcast[DenseVector[Double]]): Iterator[(DenseVector[Double], DenseVector[Double])] = {

    val zipPair = zipData.next()
    val localData = zipPair._2
    val alpha = zipPair._1
    val v = vInit.value

    val (deltaAlpha,deltaV) = localSolver.optimize(model, localData, v, alpha)

    Iterator((deltaV, deltaAlpha))
  }

  def optimize(model: Model[HingeLoss], data: RDD[LabelledPoint])
    (implicit sc: SparkContext) =
    new CoCoA(new SDCAOptimizer(new SVMOptimizer)).optimize(model, data)

  def optimize(model: Model[RidgeLoss], data: RDD[LabelledPoint])
    (implicit sc: SparkContext, d: DummyImplicit) =
    new CoCoA(new SDCAOptimizer(new RidgeOptimizer)).optimize(model, data)

  def optimize(model: Model[Loss[RealFunction,RealFunction]], data: RDD[LabelledPoint])
    (implicit sc: SparkContext, d: DummyImplicit, d2: DummyImplicit) =
    new CoCoA(new SDCAOptimizer(new BrentMethodOptimizer)).optimize(model, data)

  def optimize(model: Model[Loss[RealFunction,DifferentiableRealFunction]], data: RDD[LabelledPoint])
    (implicit sc: SparkContext, d: DummyImplicit, d2: DummyImplicit, d3: DummyImplicit) =
    new CoCoA(new SDCAOptimizer(new BrentMethodOptimizerWithFirstDerivative)).optimize(model, data)
}
