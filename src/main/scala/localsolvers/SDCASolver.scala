package localsolvers

import java.security.SecureRandom

import distopt.utils.VectorOps._
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Created by simone on 18/03/15.
 */
class SDCASolver(scOptimizer: SingleCoordinateOptimizerTrait, localIters: Int, lambda: Double, n: Long)
  extends LocalSolverTrait {
  /**
   * @param localData the local data examples
   * @param wOld ...
   * @param alphaOld ...
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
