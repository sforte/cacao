package distopt.solvers

import localsolvers.{SingleCoordinateOptimizerTrait, RealFunction, LocalSolverTrait}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import distopt.utils.Implicits._
import distopt.utils._

object CoCoA {

  /**
   * CoCoA - Communication-efficient distributed dual Coordinate Ascent.
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
  def runCoCoA(
    sc: SparkContext, 
    data: RDD[(Array[SparseClassificationPoint], LocalSolverTrait)],
    wInit: Array[Double], 
    numRounds: Int,
    beta: Double, 
    chkptIter: Int,
    n: Int,
    testData: RDD[SparseClassificationPoint],
    debugIter: Int,
    lambda: Double,
    seed: Int) : (Array[Double], RDD[Array[Double]]) = {
    
    val parts = data.partitions.size

    println("\nRunning CoCoA on "+n+" data examples, distributed over "+parts+" workers")

    var w = wInit
    val scaling = beta / parts

    var alpha = data.map(_._1.map(x => 0.0))

    for(t <- 1 to numRounds){

      val zipData = (alpha zip data).map(x => (x._1,x._2._1,x._2._2))

      val updates = zipData.mapPartitions(partitionUpdate(_,w,scaling,seed+t),preservesPartitioning=true).persist()
      alpha = updates.map(kv => kv._2)
      val primalUpdates = updates.map(kv => kv._1).reduce(_ plus _)
      w = primalUpdates.times(scaling).plus(w)

      println("Iteration: " + t)

      if(t % chkptIter == 0){
        data.checkpoint()
        alpha.checkpoint()
      }
    }

    (w, alpha)
  }

  /**
   * Performs one round of local updates using a given local dual algorithm, 
   * here locaSDCA. Will perform localIters many updates per worker.
   * @param zipData ...
   * @param wInit ...
   * @param scaling this is the scaling factor beta/K in the paper
   * @param seed ...
   * @return
   */
  private def partitionUpdate(
    zipData: Iterator[(Array[Double], Array[SparseClassificationPoint], LocalSolverTrait)],
    wInit: Array[Double],
    scaling: Double,
    seed: Int): Iterator[(Array[Double], Array[Double])] = {

    val zipPair = zipData.next()
    val localData = zipPair._2
    var alpha = zipPair._1
    val localSolver = zipPair._3
    val alphaOld = alpha.clone()

    val (deltaAlpha, deltaW) = localSolver.optimize(localData, wInit, alpha, seed)

    alpha = alphaOld.plus(deltaAlpha.times(scaling))
    Iterator((deltaW, alpha))
  }
}
