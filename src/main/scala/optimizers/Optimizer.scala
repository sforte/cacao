package optimizers

import breeze.linalg.{DenseVector, Vector}
import models.DualModel
import vectors.LabelledPoint

trait Optimizer[-ModelType<:DualModel, -DataType, AlphasType, WType] extends Serializable {

  def optimize(model: ModelType, data: DataType, alpha: AlphasType, w: WType, epsilon: Double = 0.0): (AlphasType,WType)
}

trait SingleCoordinateOptimizer[-ModelType<:DualModel] extends Optimizer[ModelType, LabelledPoint, Double, Vector[Double]]

trait LocalOptimizer[-ModelType<:DualModel] extends Optimizer[ModelType, Array[LabelledPoint], DenseVector[Double], DenseVector[Double]]