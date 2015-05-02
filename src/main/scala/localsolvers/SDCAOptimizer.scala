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
   * @param vOld the old value of v
   * @param alphaOld the old value for the alpha in the partition
   * @return deltaAlpha and deltaW, summarizing the performed local changes, see paper
   */
  override def optimize(
    model: ModelType,
    localData: Array[LabeledPoint],
    vOld: DenseVector,
    alphaOld: DenseVector,
    epsilon: Double = 0.0) : (DenseVector,DenseVector) = {

    val lambda = model.regularizer.lambda

    val alpha = alphaOld.copy
    val v = vOld.copy

    val nLocal = localData.length
    val r = new SecureRandom
    r.setSeed(0)

    // perform local udpates
    for (i <- util.Random.shuffle(1 to nLocal)) {
      // randomly select a local example
//      val idx = r.nextInt(nLocal)
      val idx = i-1

      val pt = localData(idx)
      val x = pt.features

      val (scDeltaAlpha, scDeltaV) = scOptimizer.optimize(model, pt, alpha(idx), v)

      plusEqual(v,scDeltaV,1.0)
//      plusEqual(w, x, scDeltaAlpha/(lambda*model.n))

      alpha.values(idx) += scDeltaAlpha
    }

    val deltaAlpha = plus(alpha,times(alphaOld,-1.0))
    val deltaV = plus(v,times(vOld,-1.0))

    (deltaAlpha, deltaV)
  }
}
