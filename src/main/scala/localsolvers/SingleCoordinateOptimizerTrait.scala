package localsolvers

import models.PrimalDualModel
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Abstraction for a line search method on a single coordinate.
 * To be used by the SDCASolver.
 */
trait SingleCoordinateOptimizerTrait[-ModelType<:PrimalDualModel] extends Serializable {

  def optimize(model: ModelType, n: Long, pt: LabeledPoint, alpha: Double, w: DenseVector): Double
}
