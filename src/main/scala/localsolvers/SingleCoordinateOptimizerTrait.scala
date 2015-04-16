package localsolvers

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Abstraction for a line search method on a single coordinate.
 * To be used by the SDCASolver.
 */
trait SingleCoordinateOptimizerTrait extends Serializable {
  def optimize(pt: LabeledPoint, alpha: Double, w: DenseVector): Double
}
