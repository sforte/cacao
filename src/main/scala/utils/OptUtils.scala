package distopt.utils

import distopt.utils.VectorOps._
import models.{DualModel,Model}
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object OptUtils {

  /*
    Compute average loss for the loss function defined in the model given a w vector
   */
  def computeAvgLoss(data: RDD[Array[LabeledPoint]], model: Model, w: Vector) : Double = {
    val n = data.map(_.size).reduce(_ + _)
    data.map(_.map(x => model.primalLoss(x.label, dot(x.features,w))).sum).reduce(_+_) / n
  }

  def computeDualityGap(data: RDD[Array[LabeledPoint]], model: DualModel, v: Vector, alpha: RDD[Double]): Double = {
    computePrimalObjective(data, model, model.regularizer.dualGradient(v)) -
      computeDualObjective(data, model, v, alpha.mapPartitions(x=>Iterator(new DenseVector(x.toArray))))
  }

  /*
    Compute primal objective for the loss function defined in the model and a L2 norm regularizer
   */
  def computePrimalObjective(data: RDD[Array[LabeledPoint]], model: Model, w: Vector): Double = {
    computeAvgLoss(data, model, w) + (model.regularizer.primal(w) * model.regularizer.lambda)
  }

  /*
    Compute dual objective value for the dual loss function defined in the model and a L2 norm regularizer
   */
  def computeDualObjective(data: RDD[Array[LabeledPoint]], model: DualModel, v: Vector, alpha: RDD[DenseVector]) = {

    val n = data.map(_.size).reduce(_+_)

    val lossTerm = (alpha zip data)
      .map(x => x._1.values zip x._2.map(_.label))
      .map(_.map(p => -model.dualLoss(p._2,-p._1)).sum)
      .reduce(_+_) / n

    val regularizer = -model.regularizer.dual(v) * model.regularizer.lambda

    lossTerm + regularizer
  }

  /*
    Computing average 0/1 loss
   */
  def computeClassificationError(data: RDD[LabeledPoint], w: Vector) : Double = {
    data.map(x => if(dot(x.features,w)*x.label > 0) 0.0 else 1.0).reduce(_+_) / data.count
  }

  /*
    Computing average absolute error
   */
  def computeAbsoluteError(data: RDD[LabeledPoint], w: Vector) : Double = {
    data.map(x => math.abs(dot(x.features,w) - x.label)).reduce(_+_) / data.count
  }

  /*
    Prints primal and dual objective values and the corresponding duality gap
   */
  def printSummaryStatsPrimalDual(
    algName: String, data: RDD[Array[LabeledPoint]], model: DualModel, v: Vector, alpha: RDD[DenseVector]) = {

    val w = model.regularizer.dualGradient(v)
    val objVal = computePrimalObjective(data, model, w)
    val dualObjVal = computeDualObjective(data, model, v, alpha)
    val dualityGap = objVal - dualObjVal

    println(
      algName +
      s" Objective Value: $objVal" +
      s"\n Dual Objective Value: $dualObjVal" +
      s"\n Duality Gap: $dualityGap"
    )

    dualityGap
  }

  def validateSolution(data: RDD[Array[LabeledPoint]], v: Vector, alpha: RDD[DenseVector], model: DualModel): Unit = {
    val (lambda,n) = (model.regularizer.lambda,model.n)
    val asdf = (data zip alpha).flatMap(x => (x._1 zip x._2.values)
      .map { case (LabeledPoint(_,x),a) =>  times(x, a/(lambda*n))}).reduce(plus)
    println(s"difference = ${l2norm(minus(asdf,v))}")
  }

  /*
    Prints the primal objective value
   */
  def printSummaryStats(algName: String, model: Model, data: RDD[Array[LabeledPoint]], v: Vector) =  {

    val w = model.regularizer.dualGradient(v)
    val objVal = computePrimalObjective(data, model, w)

    println(
      algName +
      s" Objective Value: $objVal"
    )
  }

  def printSummaryStatsFromW(algName: String, model: Model, data: RDD[Array[LabeledPoint]], w: Vector) =  {

    val objVal = computePrimalObjective(data, model, w)

    println(
      algName +
        s" Objective Value: $objVal"
    )
  }
}
