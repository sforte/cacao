package distopt.solvers

import localsolvers.RealFunction
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import distopt.utils.Implicits._
import distopt.utils._

object MinibatchCD {

  /**
   * Mini-batch SDCA, that is dual coordinate descent for standard hinge-loss SVM
   * 
   * @param sc
   * @param data RDD of all data examples
   * @param wInit initial weight vector (has to be zero)
   * @param numRounds number of outer iterations T in the paper
   * @param localIters number of inner localSDCA iterations, H in the paper
   * @param lambda the regularization parameter
   * @param beta scaling parameter. beta=1 gives averaging, beta=K=data.partitions.size gives (aggressive) adding
   * @param chkptIter checkpointing the resulting RDDs from time to time, to ensure persistence and shorter dependencies
   * @param testData
   * @param debugIter
   * @param seed
   * @return
   */
  def runMbCD(
    sc: SparkContext, 
    data: RDD[SparseClassificationPoint],
    n: Int,
    wInit: Array[Double], 
    numRounds: Int, 
    localIters: Int, 
    lambda: Double, 
    beta: Double, 
    chkptIter: Int, 
    testData: RDD[SparseClassificationPoint], 
    debugIter: Int,
    seed: Int) : (Array[Double], RDD[Array[Double]]) = {
    
    val parts = data.partitions.size 	// number of partitions of the data, K in the paper
    println("\nRunning Mini-batch CD on "+n+" data examples, distributed over "+parts+" workers")
    
    // initialize alpha, w
    var alphaVars = data.map(x => 0.0).cache()
    var alpha = alphaVars.mapPartitions(x => Iterator(x.toArray))
    var dataArr = data.mapPartitions(x => Iterator(x.toArray))
    var w = wInit
    val scaling = beta / (parts * localIters);

    for(t <- 1 to numRounds){

      // zip alpha with data
      val zipData = alpha.zip(dataArr)

      // find updates to alpha, w
      val updates = zipData.mapPartitions(partitionUpdate(_,w,localIters,lambda,n,scaling,seed+t),preservesPartitioning=true).persist()
      alpha = updates.map(kv => kv._2)
      val primalUpdates = updates.map(kv => kv._1).reduce(_ plus _)
      w = primalUpdates.times(scaling).plus(w)

      // optionally calculate errors
      if (debugIter>0 && t % debugIter == 0) {
//        println("Iteration: " + t)
//        println("primal objective: " + OptUtils.computePrimalObjective(data, w, lambda, primalLosses))
//        println("primal-dual gap: " + OptUtils.computeDualityGap(data, w, alpha, lambda, primalLosses, dualLosses))
        if (testData != null) { println("test error: " + OptUtils.computeClassificationError(testData, w)) }
      }

      // optionally checkpoint RDDs
      if(t % chkptIter == 0){
        zipData.checkpoint()
        alpha.checkpoint()
      }
    }

    return (w, alpha)
  }

  /**
   * Performs one round of mini-batch CD updates
   *
   * @param zipData
   * @param localIters
   * @param lambda
   * @param n
   * @param scaling - possible scaling beta/(K*H) in the paper
   * @param seed
   * @return
   */
  private def partitionUpdate(
    zipData: Iterator[(Array[Double],Array[SparseClassificationPoint])],//((Int, Double), SparseClassificationPoint)],
    wInit: Array[Double], 
    localIters: Int, 
    lambda: Double, 
    n: Int, 
    scaling: Double,
    seed: Int): Iterator[(Array[Double], Array[Double])] = {

    val zipPair = zipData.next()
    var localData = zipPair._2
    var alpha = zipPair._1
    val alphaOld = alpha.clone
    var w = wInit
    val nLocal = localData.length
    var r = new scala.util.Random(seed)
    var deltaW = Array.fill(wInit.length)(0.0)

        // perform local udpates
    for (i <- 1 to localIters) {

      // randomly select a local example
      val idx = r.nextInt(nLocal)
      val currPt = localData(idx)
      var y = currPt.label
      val x = currPt.features

      // compute hinge loss gradient
      val grad = (y*(x.dot(w)) - 1.0)*(lambda*n)

      // compute projected gradient
      var proj_grad = grad
      if (alpha(idx) <= 0.0)
        proj_grad = Math.min(grad,0)
      else if (alpha(idx) >= 1.0)
        proj_grad = Math.max(grad,0)

      if (Math.abs(proj_grad) != 0.0 ) {
        val qii  = x.dot(x)
        var newAlpha = 1.0
        if (qii != 0.0) {
          newAlpha = Math.min(Math.max((alpha(idx) - (grad / qii)), 0.0), 1.0)
        }

        // update primal and dual variables
        val update = x.times( y*(newAlpha-alpha(idx))/(lambda*n) )
        deltaW = update.plus(deltaW)
        alpha(idx) = newAlpha
      }
    }

    val deltaAlpha = (alphaOld.times(-1.0)).plus(alpha)   
    alpha = alphaOld.plus(deltaAlpha.times(scaling))

    // return weight vector, alphas
    return Iterator((deltaW, alpha))
  }

}
