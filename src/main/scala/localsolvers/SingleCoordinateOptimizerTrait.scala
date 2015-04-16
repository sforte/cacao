package localsolvers

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Created by simone on 18/03/15.
 */
trait SingleCoordinateOptimizerTrait extends Serializable {
  def optimize(pt: LabeledPoint, alpha: Double, w: DenseVector): Double
}
