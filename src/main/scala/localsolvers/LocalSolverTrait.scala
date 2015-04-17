package localsolvers

import models.PrimalDualModel
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * An abstraction for a local dual solver to be used by CoCoA
 */
trait LocalSolverTrait [-ModelType<:PrimalDualModel] extends Serializable {

  /**
   * @param seed ...
   * @return
   */
  def optimize(model: ModelType, n: Long, localData: Array[LabeledPoint],
               wOld: DenseVector, alphaOld: DenseVector, seed : Int = 0): (DenseVector,DenseVector)
}
