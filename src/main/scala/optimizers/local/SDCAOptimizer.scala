package optimizers.local

import java.security.SecureRandom
import breeze.linalg.DenseVector
import models.{Regularizer, RealFunction, Loss}
import optimizers.{LocalOptimizer, SingleCoordinateOptimizer}
import vectors.LabelledPoint

/**
 * A Single Dual Coordinate Ascent based local solver
 * @param scOptimizer The method used to optimize on a single coordinate
 */
class SDCAOptimizer [-LossType<:Loss[RealFunction,RealFunction]]
  (scOptimizer: SingleCoordinateOptimizer[LossType], numPasses: Int)
  extends LocalOptimizer[LossType] {

  /**
   * @param localData the local data examples
   * @param vOld the old value of v
   * @param alphaOld the old value for the alpha in the partition
   * @return deltaAlpha and deltaW, summarizing the performed local changes, see paper
   */
  override def optimize(
    model: LossType,
    regularizer: Regularizer,
    n: Long,
    localData: Array[LabelledPoint],
    vOld: DenseVector[Double],
    alphaOld: DenseVector[Double],
    epsilon: Double = 0.0) : (DenseVector[Double],DenseVector[Double]) = {

    val alpha = alphaOld.copy
    val v = vOld.copy

    val nLocal = localData.length
    val r = new SecureRandom
    r.setSeed(0)

    for (pass <- 1 to numPasses) {

      val randperm = util.Random.shuffle(1 to nLocal)

      for (i <- randperm; idx = i-1) {

        val pt = localData(idx)

        val (scDeltaAlpha, scDeltaV) =
          scOptimizer.optimize(model, regularizer, n, pt, alpha(idx), v)

        v += scDeltaV
        alpha(idx) += scDeltaAlpha
      }
    }

    val deltaAlpha = alpha - alphaOld
    val deltaV = v - vOld

    (deltaAlpha, deltaV)
  }
}