package models

import breeze.linalg.{DenseVector, SparseVector, Vector}
import vectors.LazyMappedVector
import vectors.VectorOps._
import math._

/*
  A function that maps a label y and a value x to a loss value.
  Used both to represent a primal and a dual loss.
 */
trait Loss extends ((Double,Double) => Double) with Serializable {
  def domain(y: Double) : (Double,Double)
}

trait DifferentiableLoss extends Loss {
  def derivative : Loss
}

trait DoublyDifferentiableLoss extends DifferentiableLoss {
  def derivative : DifferentiableLoss
}

trait MultivariateFunction extends (Vector[Double] => Double) with Serializable
trait Gradient extends (Vector[Double] => Vector[Double]) with Serializable

trait Regularizer extends Serializable {
  def primal(w: Vector[Double]): Double
  def dualGradient(w: Vector[Double]): Vector[Double]
  def lambda: Double

  def dual(v: Vector[Double]) = {
    val w = dualGradient(v)
    (w dot v) - primal(w)
  }
}

trait Model extends Serializable {
  def primalLoss: Loss
  var regularizer: Regularizer
  def n: Long
}

/*
  Class representing a classification/regression model as defined in the CoCoA paper.
 */
trait DualModel extends Model {
  def primalLoss: Loss
  def dualLoss: Loss
}

/*
Class representing a classification/regression model with a differentiable dual loss.
*/
trait DualModelWithFirstDerivative extends DualModel {
  override def dualLoss: DifferentiableLoss
}

/*
Class representing a classification/regression model with a doubly differentiable dual loss.
*/
trait DualModelWithSecondDerivative extends DualModelWithFirstDerivative {
  override def dualLoss: DoublyDifferentiableLoss
}



