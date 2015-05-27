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
class PasscodeWild [-LossType<:Loss[RealFunction,RealFunction]]
  (scOptimizer: SingleCoordinateOptimizer[LossType], numPasses: Int, numThreads: Int)
    extends LocalOptimizer[LossType] {

  /**
   * @param localData the local data examples
   * @param vOld the old value of v
   * @param alphaOld the old value for the alpha in the partition
   * @return deltaAlpha and deltaW, summarizing the performed local changes, see paper
   */
  override def optimize(
    loss: LossType,
    regularizer: Regularizer,
    n: Long,
    localData: Array[LabelledPoint],
    vOld: DenseVector[Double],
    alphaOld: DenseVector[Double]):
  (DenseVector[Double],DenseVector[Double]) = {

    val alpha = alphaOld.copy
    val v = vOld.copy

    val nLocal = localData.length
    val r = new SecureRandom
    r.setSeed(0)

    (1 to numThreads).par.map { _ =>

      val randperm = util.Random.shuffle(1 to nLocal)

      for (pass <- 1 to numPasses) {

        for (i <- randperm; idx = i - 1) {
          val pt = localData(idx)
          val a = alpha(idx)

          val (scDeltaAlpha, scDeltaV) =
            scOptimizer.optimize(loss, regularizer, n, pt, a, v)

          alpha(idx) += scDeltaAlpha
          v += scDeltaV
        }
      }
    }

    val deltaAlpha = alpha - alphaOld
    val deltaV = ((alpha.valuesIterator.toArray zip localData).par.map { case (alpha, LabelledPoint(y,x)) =>
      x * (alpha / (regularizer.lambda*n))
    }.reduce(_+_) - vOld).asInstanceOf[DenseVector[Double]]

    (deltaAlpha, deltaV)
  }
}
