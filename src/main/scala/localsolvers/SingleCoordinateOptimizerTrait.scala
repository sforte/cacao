package localsolvers

import distopt.utils.SparseClassificationPoint

/**
 * Created by simone on 18/03/15.
 */
trait SingleCoordinateOptimizerTrait extends Serializable {
  def optimize(pt: SparseClassificationPoint, alpha: Double, w: Array[Double]): Double
}
