package distopt.utils

import localsolvers.RealFunction
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scala.util._
import java.io._
import scala.io.Source

object OptUtils {

  // load data stored in LIBSVM format and append line ID
  def loadLIBSVMData(sc: SparkContext, filename: String, numSplits: Int, numFeats: Int): RDD[SparseClassificationPoint] = {

    // read in text file

    val data = sc.textFile(filename,numSplits).repartition(numSplits)
    val numEx = data.count()

    // find number of elements per partition
    val numParts = data.partitions.size
    val sizes = data.mapPartitionsWithSplit{ case(i,lines) =>
      Iterator(i -> lines.length)
    }.collect().sortBy(_._1)
    sizes.foreach(println)
    val offsets = sizes.map(x => x._2).scanLeft(0)(_+_).toArray

    // parse input
    data.mapPartitionsWithSplit { case(partition, lines) =>
      lines.zipWithIndex.flatMap{ case(line, idx) =>

        // calculate index for line
        val index = offsets(partition) + idx

        if(index < numEx){

          // parse label
          val parts = line.trim().split(' ')

          val label = parts(0).toDouble
//          var label = -1
//          if (parts(0).contains("+") || parts(0).toInt == 1)
//            label = 1

          // parse features
          val featureArray = parts.slice(1,parts.length)
            .map(_.split(':') 
            match { case Array(i,j) => (i.toInt-1,j.toDouble)}).filter(_._2 != 0).toArray
          var features = new SparseVector(featureArray.map(x=>x._1), featureArray.map(x=>x._2))

          // create classification point
          Iterator(SparseClassificationPoint(index,label,features))
        }
        else{
          Iterator()
        }
      }
    }
      .map(x => SparseClassificationPoint(x.index,x.label,
      new SparseVector(
        x.features.indices.filter(_ >= 2),
        (x.features.indices zip x.features.values).filter(_._1 >= 2).map(_._2))
      ))
  }

  def loadImageNetData(sc: SparkContext, filename: String, nsplits: Int): RDD[SparseClassificationPoint] = {

    // read in text file
    val data = sc.textFile(filename,nsplits)
    val c = data.first.split(",")(0)

    // find number of elements per partition
    val numParts = data.partitions.size
    val sizes = data.mapPartitionsWithSplit{ case(i,lines) =>
      Iterator(i -> lines.length)
    }.collect().toMap
    val offsets = sizes.values.scanLeft(0)(_+_).toArray

    // parse input
    data.mapPartitionsWithSplit { case(partition, lines) =>
      lines.zipWithIndex.map{ case(line, idx) =>

        // calculate index for line
        val index = offsets(partition) + idx

        // parse label
        val parts = line.trim().split(",")
        val label = if(parts(0)==c) 1 else -1

        // parse features
        val featureArray = (1 to parts.length-1)
          .map( a => (a.toInt-1,parts(a).toDouble)).filter{ case(a,b) => b!=0.0 }.toArray
        val features = new SparseVector(featureArray.map(x => x._1), featureArray.map(x => x._2))

        // create classification point
        SparseClassificationPoint(index,label,features)
      }
    }

  }

  // dot product of two dense arrays
  def dotDense(dense1: Array[Double], dense2: Array[Double]) : Double = {
    return dense1.zipWithIndex.map{ case(v,i) => v*dense2(i) }.sum
  }

  // find norm of dense arrays
  def normDense(dense: Array[Double]): Double = {
    dotDense(dense,dense)
  }

  // calculate hinge loss for a point (label,features) given a weight vector
  def hingeLoss(point: SparseClassificationPoint, w:Array[Double]) : Double = {
    val y = point.label
    val X = point.features
    return Math.max(1 - y*(X.dot(w)),0.0)
  }

  // compute hinge gradient for a point (label, features) given a weight vector
  def computeHingeGradient(point: SparseClassificationPoint, w: Array[Double]) : SparseVector = {
    val y = point.label
    val X = point.features
    val eval = 1 - y * (X.dot(w))
    if(eval > 0){
      return X.times(y)
    }
    else{
      return new SparseVector(Array[Int](),Array[Double]())
    }
  }

  // compute dual loss
  def dualLoss(d:(SparseClassificationPoint,SparseClassificationPoint), a:Array[Double]) : Double = {
    val qij = d._1.features.dot(d._2.features)
    return .5*a(d._1.index)*a(d._2.index)*d._1.label*d._2.label*qij
  }

  // can be used to compute train or test error
  def computeAvgLoss(data: RDD[SparseClassificationPoint], w: Array[Double], losses: RDD[RealFunction]) : Double = {
    val n = data.count()
    return (data zip losses).map(x => x._2(x._1.features.dot(w))).reduce(_+_) / n
  }

  /**
   * Compute the primal objective function value.
   * Caution:just use for debugging purposes. this is an expensive operation, taking one full pass through the data
   * 
   * @param data
   * @param w
   * @param lambda
   * @return
   */
  def computePrimalObjective(
      data: RDD[SparseClassificationPoint], w: Array[Double], lambda: Double, loss: RDD[RealFunction]): Double = {
    return (computeAvgLoss(data, w, loss) + (normDense(w) * lambda * 0.5))
  }

  /**
   * Compute the dual objective function value.
   * Caution:just use for debugging purposes. this is an expensive operation, taking one full pass through the data
   *
   * @param w
   * @param lambda
   * @return
   */
  def computeDualObjective(
    w: Array[Double], alpha : RDD[Array[Double]], lambda: Double, losses: RDD[RealFunction]): Double = {

    val n = alpha.map(_.size).reduce(_ + _)

    val asdf = (alpha zip losses.mapPartitions(x => Iterator(x.toArray)))
      .map( x => (x._1 zip x._2).map{ case(x,loss) => loss(-x) }.sum ).reduce(_ + _) / n
//println(asdf)
    -lambda / 2 * OptUtils.normDense(w) - asdf
  }
  /**
   * Compute the duality gap value.
   * Caution:just use for debugging purposes. this is an expensive operation, taking one full pass through the data
   *
   * @param data
   * @param w
   * @param alpha
   * @param lambda
   * @return
   */
  def computeDualityGap(data: RDD[SparseClassificationPoint], w: Array[Double],
                          alpha: RDD[Array[Double]], lambda: Double,
                          primalLoss: RDD[RealFunction], dualLoss: RDD[RealFunction]): Double = {
    return (computePrimalObjective(data, w, lambda, primalLoss) - computeDualObjective(w, alpha, lambda, dualLoss))
  }

  def computeClassificationError(data: RDD[SparseClassificationPoint], w:Array[Double]) : Double = {
    val n = data.count()
    return data.map(x => if((x.features).dot(w)*(x.label) > 0) 0.0 else 1.0).reduce(_ + _)/n
  }

  def computeAbsoluteError(data: RDD[SparseClassificationPoint], w:Array[Double]) : Double = {
    val n = data.count
    println(n)
    data.map(x => (x.features.dot(w),x.label)).take(100).foreach(x=>println(s"${x._1} ${x._2}"))
    return data.map(x => math.abs(x.features.dot(w) - x.label)).reduce(_ + _)/n
    //    println(w.mkString(" "))
    //    return data.map(x => x.features.dot(w) - x.label).map(x => x*x).reduce(_ + _) / n
  }

  def printSummaryStatsPrimalDual(algName: String, data: RDD[SparseClassificationPoint], w: Array[Double], alpha: RDD[Array[Double]],
        lambda: Double, testData: RDD[SparseClassificationPoint], primalLosses: RDD[RealFunction], dualLosses: RDD[RealFunction]) = {
    var outString = algName + " has finished running. Summary Stats: "
    val objVal = computePrimalObjective(data, w, lambda, primalLosses)
    outString = outString + "\n Total Objective Value: " + objVal
    val dualityGap = computeDualityGap(data, w, alpha, lambda, primalLosses, dualLosses)
    outString = outString + "\n Duality Gap: " + dualityGap
    if(testData!=null){
//      val testErr = computeClassificationError(testData, w)
      val testErr = computeAbsoluteError(testData, w)
      outString = outString + "\n Test Error: " + testErr
    }
    println(outString + "\n")
  }

  def printSummaryStats(algName: String, data: RDD[SparseClassificationPoint], w: Array[Double],
      lambda: Double, testData: RDD[SparseClassificationPoint], losses: RDD[RealFunction]) =  {
    var outString = algName + " has finished running. Summary Stats: "
    val objVal = computePrimalObjective(data, w, lambda, losses)
    outString = outString + "\n Total Objective Value: " + objVal
    if(testData!=null){
      val testErr = computeClassificationError(testData, w)
      outString = outString + "\n Test Error: " + testErr
    }
    println(outString + "\n")
  }
  
}
