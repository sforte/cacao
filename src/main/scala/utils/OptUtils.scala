package utils

import breeze.linalg.{DenseVector, Vector}
import models.{DualModel, Model}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import vectors.VectorOps._
import vectors.{LabelledPoint, VectorOps}

object OptUtils {

  /*
    Compute average loss for the loss function defined in the model given a w vector
   */
  def computeAvgLoss(data: RDD[Array[LabelledPoint]], model: Model, w: Vector[Double]) : Double = {
    val n = data.map(_.size).reduce(_ + _)
    data.map(_.map(x => model.primalLoss(x.label, x.features dot w)).sum).reduce(_+_) / n
  }

  def computeDualityGap(data: RDD[Array[LabelledPoint]], model: DualModel, v: Vector[Double], alpha: RDD[Double]): Double = {
    computePrimalObjective(data, model, model.regularizer.dualGradient(v)) -
      computeDualObjective(data, model, v, alpha.mapPartitions(x=>Iterator(new DenseVector(x.toArray))))
  }

  /*
    Compute primal objective for the loss function defined in the model and a L2 norm regularizer
   */
  def computePrimalObjective(data: RDD[Array[LabelledPoint]], model: Model, w: Vector[Double]): Double = {
    computeAvgLoss(data, model, w) + (model.regularizer.primal(w) * model.regularizer.lambda)
  }

  /*
    Compute dual objective value for the dual loss function defined in the model and a L2 norm regularizer
   */
  def computeDualObjective(data: RDD[Array[LabelledPoint]], model: DualModel, v: Vector[Double], alpha: RDD[DenseVector[Double]]) : Double = {

    val n = data.map(_.size).reduce(_+_)

    val lossTerm = (alpha zip data)
      .map(x => x._1.data zip x._2.map(_.label))
      .map(_.map(p => -model.dualLoss(p._2,-p._1)).sum)
      .reduce(_+_) / n

    val regularizer = -model.regularizer.dual(v) * model.regularizer.lambda

    lossTerm + regularizer
  }

  /*
    Computing average 0/1 loss
   */
  def computeClassificationError(data: RDD[LabelledPoint], w: Vector[Double]) : Double = {
    data.map(x => if((x.features dot w)*x.label > 0) 0.0 else 1.0).reduce(_+_) / data.count
  }

  /*
    Computing average absolute error
   */
  def computeAbsoluteError(data: RDD[LabelledPoint], w: Vector[Double]) : Double = {
    data.map(x => math.abs((x.features dot w) - x.label)).reduce(_+_) / data.count
  }

  /*
    Prints primal and dual objective values and the corresponding duality gap
   */
  def printSummaryStatsPrimalDual(
    algName: String, data: RDD[Array[LabelledPoint]], model: DualModel, v: Vector[Double], alpha: RDD[DenseVector[Double]]) = {

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

  def validateSolution(data: RDD[Array[LabelledPoint]], v: Vector[Double], alpha: RDD[DenseVector[Double]], model: DualModel): Unit = {
    val (lambda,n) = (model.regularizer.lambda,model.n)

    val asdf = (data zip alpha).flatMap(x => (x._1 zip x._2.data)
      .map { case (LabelledPoint(_,x),a) =>  x * (a/(lambda*n))}).reduce(_+_)
    println(s"difference = ${l2norm(asdf - v)}")
  }

  /*
    Prints the primal objective value
   */
  def printSummaryStats(algName: String, model: Model, data: RDD[Array[LabelledPoint]], v: Vector[Double]) =  {

    val w = model.regularizer.dualGradient(v)
    val objVal = computePrimalObjective(data, model, w)

    println(
      algName +
      s" Objective Value: $objVal"
    )
  }

  def printSummaryStatsFromW(algName: String, model: Model, data: RDD[Array[LabelledPoint]], w: Vector[Double]) =  {

    val objVal = computePrimalObjective(data, model, w)

    println(
      algName +
        s" Objective Value: $objVal"
    )
  }

  def loadLibSVMFile(sc: SparkContext, trainFile: String, numFeatures: Int, numSplits: Int) = {
    MLUtils.loadLibSVMFile(sc, trainFile, numFeatures, numSplits).map(p =>
      new LabelledPoint(p.label, VectorOps.toBreeze(p.features)))
  }

  def time(label: String) (func: => Unit) = {
    val start = System.currentTimeMillis()
    func
    val end = System.currentTimeMillis()
    println(s"$label: ${(end-start)/1000.0}")
  }

  def loadData(numFeatures: Int, file: String) = {
    val conf = new SparkConf().setMaster("local[4]")
      .setAppName("demoCoCoA")
      .setJars(SparkContext.jarOfObject(this).toSeq)
    val sc = new SparkContext(conf)

    val file = "data/trainset.dat"
    val numFeatures = 100000

    OptUtils.loadLibSVMFile(sc, file, numFeatures, 1)
  }
}
