package localsolvers

import java.security.SecureRandom

import distopt.utils.VectorOps._
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * A Single Dual Coordinate Ascent based local solver
 * @param scOptimizer The method used to optimize on a single coordinate
 * @param localIters Number of iterations we run the method locally
 * @param lambda Lambda parameter of the model
 * @param n Number of data points
 */
class SDCASolver(scOptimizer: SingleCoordinateOptimizerTrait, localIters: Int, lambda: Double, n: Long)
  extends LocalSolverTrait {

  /**
   * @param localData the local data examples
   * @param wOld the old value of w
   * @param alphaOld the old value for the alpha in the partition
   * @param seed ...
   * @return deltaAlpha and deltaW, summarizing the performed local changes, see paper
   */
  override def optimize(
    localData: Array[LabeledPoint],
    wOld: DenseVector,
    alphaOld: DenseVector,
    seed : Int = 0) : (DenseVector,DenseVector) = {

    val alpha = alphaOld.copy
    val w = wOld.copy

    val nLocal = localData.length
    val r = new SecureRandom
    r.setSeed(seed)

    // perform local udpates
    for (i <- 1 to localIters) {
      // randomly select a local example
      val idx = r.nextInt(nLocal)

      val pt = localData(idx)
      val x = pt.features

      val scDeltaAlpha = scOptimizer.optimize(pt, alpha(idx), w)

      plusEqual(w,x,scDeltaAlpha/(lambda*n))
      alpha.values(idx) += scDeltaAlpha
    }

    val deltaAlpha = plus(alpha,times(alphaOld,-1.0))
    val deltaW = plus(w,times(wOld,-1.0))

    (deltaAlpha, deltaW)
  }
}
