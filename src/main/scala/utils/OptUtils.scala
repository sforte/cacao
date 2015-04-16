package distopt.utils

import distopt.utils.VectorOps._
import models.PrimalDualModel
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object OptUtils {

  def computeAvgLoss(data: RDD[Array[LabeledPoint]], model: PrimalDualModel, w: Vector) : Double = {
    val n = data.map(_.size).reduce(_ + _)
    data.map(_.map(x => model.primalLoss(x.label, dot(x.features,w))).sum).reduce(_+_) / n
  }

  def computePrimalObjective(data: RDD[Array[LabeledPoint]], model: PrimalDualModel, w: Vector): Double = {
    computeAvgLoss(data, model, w) + (dot(w,w) * model.lambda * 0.5)
  }

  def computeDualObjective(data: RDD[Array[LabeledPoint]], model: PrimalDualModel, w: Vector, alpha: RDD[DenseVector]) = {

    val n = data.map(_.size).reduce(_+_)

    val lossTerm = (alpha zip data)
      .map(x => x._1.values zip x._2.map(_.label))
      .map(_.map(p => -model.dualLoss(p._2,-p._1)).sum)
      .reduce(_+_) / n

    val regularizer = -dot(w,w) * model.lambda * 0.5

    lossTerm + regularizer
  }

  def computeDualityGap(data: RDD[Array[LabeledPoint]], model: PrimalDualModel, w: DenseVector, alpha: RDD[DenseVector]) = {
    computePrimalObjective(data, model, w) - computeDualObjective(data, model, w, alpha)
  }

  def computeClassificationError(data: RDD[LabeledPoint], w: Vector) : Double = {
    data.map(x => if(dot(x.features,w)*x.label > 0) 0.0 else 1.0).reduce(_+_) / data.count
  }

  def computeAbsoluteError(data: RDD[LabeledPoint], w: Vector) : Double = {
    data.map(x => math.abs(dot(x.features,w) - x.label)).reduce(_+_) / data.count
  }

  def printSummaryStatsPrimalDual(
    algName: String, data: RDD[Array[LabeledPoint]], model: PrimalDualModel, w: Vector, alpha: RDD[DenseVector]) {

    val objVal = computePrimalObjective(data, model, w)
    val dualObjVal = computeDualObjective(data, model, w, alpha)
    val dualityGap = objVal - dualObjVal

    println(
      s"$algName has finished running. Summary Stats: " +
      s"\n Objective Value: $objVal" +
      s"\n Dual Objective Value: $dualObjVal" +
      s"\n Duality Gap: $dualityGap"
    )

//    if(testData!=null && false) {
//      val testErr = computeClassificationError(testData, w)
//      outString = outString + "\n Test Error: " + testErr
//    }
  }

  def printSummaryStats(algName: String, model: PrimalDualModel, data: RDD[Array[LabeledPoint]], w: Vector) =  {

    val objVal = computePrimalObjective(data, model, w)

    println(
      s"$algName has finished running. Summary Stats: " +
        s"\n Objective Value: $objVal"
    )

//    if(testData!=null){
//      val testErr = computeClassificationError(testData, w)
//      outString = outString + "\n Test Error: " + testErr
//    }
  }
  
}
