package localsolvers

import java.security.SecureRandom

import distopt.utils.VectorOps._
import models.DualModel
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * A Single Dual Coordinate Ascent based local solver
 * @param scOptimizer The method used to optimize on a single coordinate
 * @param localIters Number of iterations we run the method locally
 */
class SDCAOptimizer [-ModelType<:DualModel] (scOptimizer: SingleCoordinateOptimizer[ModelType], localIters: Int)
  extends LocalOptimizer[ModelType] {

  /**
   * @param localData the local data examples
   * @param wOld the old value of w
   * @param alphaOld the old value for the alpha in the partition
   * @return deltaAlpha and deltaW, summarizing the performed local changes, see paper
   */
  override def optimize(
    model: ModelType,
    localData: Array[LabeledPoint],
    wOld: DenseVector,
    alphaOld: DenseVector) : (DenseVector,DenseVector) = {

    val lambda = model.lambda

    val alpha = alphaOld.copy
    val w = wOld.copy

    val nLocal = localData.length
    val r = new SecureRandom
    r.setSeed(0)

    // perform local udpates
    for (i <- 1 to localIters) {
      // randomly select a local example
      val idx = r.nextInt(nLocal)

      val pt = localData(idx)
      val x = pt.features

      val (scDeltaAlpha, scDeltaW) = scOptimizer.optimize(model, pt, alpha(idx), w)

      plusEqual(w,scDeltaW,1.0)
//      plusEqual(w, x, scDeltaAlpha/(lambda*model.n))

      alpha.values(idx) += scDeltaAlpha
    }

    val deltaAlpha = plus(alpha,times(alphaOld,-1.0))
    val deltaW = plus(w,times(wOld,-1.0))

    (deltaAlpha, deltaW)
  }
}
