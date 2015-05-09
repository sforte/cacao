package optimizers

import breeze.linalg.{DenseVector, Vector}
import models.{Regularizer, RealFunction, Loss}
import vectors.LabelledPoint

trait Optimizer[-LossType<:Loss[RealFunction,RealFunction], -DataType, AlphasType, WType] extends Serializable {

  def optimize(loss: LossType, reg: Regularizer, n: Long,
               data: DataType, alpha: AlphasType, w: WType, epsilon: Double = 0.0): (AlphasType,WType)
}

trait SingleCoordinateOptimizer[-LossType<:Loss[RealFunction,RealFunction]]
  extends Optimizer[LossType, LabelledPoint, Double, Vector[Double]]

trait LocalOptimizer[-LossType<:Loss[RealFunction,RealFunction]]
  extends Optimizer[LossType, Array[LabelledPoint], DenseVector[Double], DenseVector[Double]]