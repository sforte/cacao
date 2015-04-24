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

  /*
    Compute primal objective for the loss function defined in the model and a L2 norm regularizer
   */
  def computePrimalObjective(data: RDD[Array[LabeledPoint]], model: Model, w: Vector): Double = {
    computeAvgLoss(data, model, w) + (dot(w,w) * model.lambda * 0.5)
  }

  /*
    Compute dual objective value for the dual loss function defined in the model and a L2 norm regularizer
   */
  def computeDualObjective(data: RDD[Array[LabeledPoint]], model: DualModel, w: Vector, alpha: RDD[DenseVector]) = {

    val n = data.map(_.size).reduce(_+_)

    val lossTerm = (alpha zip data)
      .map(x => x._1.values zip x._2.map(_.label))
      .map(_.map(p => -model.dualLoss(p._2,-p._1)).sum)
      .reduce(_+_) / n

    val regularizer = -dot(w,w) * model.lambda * 0.5

    lossTerm + regularizer
  }

  /*
    Different between primal objective and dual objective value
   */
  def computeDualityGap(data: RDD[Array[LabeledPoint]], model: DualModel, w: DenseVector, alpha: RDD[DenseVector]) = {
    computePrimalObjective(data, model, w) - computeDualObjective(data, model, w, alpha)
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
    algName: String, data: RDD[Array[LabeledPoint]], model: DualModel, w: Vector, alpha: RDD[DenseVector]) {

    val objVal = computePrimalObjective(data, model, w)
    val dualObjVal = computeDualObjective(data, model, w, alpha)
    val dualityGap = objVal - dualObjVal

    println(
      s" Objective Value: $objVal" +
      s"\n Dual Objective Value: $dualObjVal" +
      s"\n Duality Gap: $dualityGap"
    )
  }

  /*
    Prints the primal objective value
   */
  def printSummaryStats(algName: String, model: Model, data: RDD[Array[LabeledPoint]], w: Vector) =  {

    val objVal = computePrimalObjective(data, model, w)

    println(s" Objective Value: $objVal")
  }
  
}
