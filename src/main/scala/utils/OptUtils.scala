package utils

import breeze.linalg.{DenseVector, Vector}
import models.{Regularizer, Loss, RealFunction}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import vectors.VectorOps._
import vectors.{LabelledPoint, VectorOps}

object OptUtils {

  /*
    Compute average loss for the loss function defined in the model given a w vector
   */
  def computeAvgLoss(data: RDD[Array[LabelledPoint]], loss: Loss[RealFunction,_], n: Long, w: Vector[Double]) = {
    data.map(_.map(x => loss(x.label)(x.features dot w)).sum).reduce(_+_) / n
  }

  def computeDualityGap(data: RDD[Array[LabelledPoint]], loss: Loss[RealFunction,RealFunction],
                        regularizer: Regularizer, n: Long, v: Vector[Double], alpha: RDD[Double]) = {
    computePrimalObjective(data, loss, regularizer, n, regularizer.dualGradient(v)) -
      computeDualObjective(data, loss, regularizer, n, v, alpha.mapPartitions(x=>Iterator(new DenseVector(x.toArray))))
  }

  /*
    Compute primal objective for the loss function defined in the model and a L2 norm regularizer
   */
  def computePrimalObjective(data: RDD[Array[LabelledPoint]], loss: Loss[RealFunction,_],
                             regularizer: Regularizer, n: Long, w: Vector[Double]) = {
    computeAvgLoss(data, loss, n, w) + regularizer.primal(w) * regularizer.lambda
  }

  /*
    Compute dual objective value for the dual loss function defined in the model and a L2 norm regularizer
   */
  def computeDualObjective(data: RDD[Array[LabelledPoint]], loss: Loss[_,RealFunction],
                           regularizer: Regularizer, n: Long,
                           v: Vector[Double], alpha: RDD[DenseVector[Double]]) = {

    val lossTerm = (alpha zip data)
      .map(x => x._1.data zip x._2.map(_.label))
      .map(_.map(p => -loss.conjugate(p._2)(-p._1)).sum)
      .reduce(_+_) / n

    val regTerm = -regularizer.dual(v) * regularizer.lambda

    lossTerm + regTerm
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
    algName: String, data: RDD[Array[LabelledPoint]], model: Loss[RealFunction,RealFunction],
    regularizer: Regularizer, n: Long,
    v: Vector[Double], alpha: RDD[DenseVector[Double]]) = {

    val w = regularizer.dualGradient(v)
    val objVal = computePrimalObjective(data, model, regularizer, n, w)
    val dualObjVal = computeDualObjective(data, model, regularizer, n, v, alpha)
    val dualityGap = objVal - dualObjVal

    println(
      algName +
      s" Objective Value: $objVal" +
      s"\n Dual Objective Value: $dualObjVal" +
      s"\n Duality Gap: $dualityGap"
    )

    dualityGap
  }

  def validateSolution(data: RDD[Array[LabelledPoint]], v: Vector[Double], alpha: RDD[DenseVector[Double]],
                       loss: Loss[RealFunction,RealFunction], regularizer: Regularizer, n: Long){
    val lambda = regularizer.lambda

    val asdf = (data zip alpha).flatMap(x => (x._1 zip x._2.data)
      .map { case (LabelledPoint(_,x),a) =>  x * (a/(lambda*n))}).reduce(_+_)
    println(s"difference = ${l2norm(asdf - v)}")
  }

  /*
    Prints the primal objective value
   */
  def printSummaryStats(algName: String, loss: Loss[RealFunction,RealFunction],
                        regularizer: Regularizer, n: Long,
                        data: RDD[Array[LabelledPoint]], v: Vector[Double]) =  {

    val w = regularizer.dualGradient(v)
    val objVal = computePrimalObjective(data, loss, regularizer, n, w)

    println(
      algName +
      s" Objective Value: $objVal"
    )
  }

  def printSummaryStatsFromW(algName: String, loss: Loss[RealFunction,RealFunction],
                             regularizer: Regularizer, n: Long,
                             data: RDD[Array[LabelledPoint]], w: Vector[Double]) =  {

    val objVal = computePrimalObjective(data, loss, regularizer, n, w)

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
    val start = System.nanoTime()
    func
    val end = System.nanoTime()
    println(s"$label: ${(end-start)/1000000000.0}")
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
