package models

import distopt.utils.VectorOps._
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}


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

trait MultivariateFunction extends (Vector => Double) with Serializable
trait Gradient extends (Vector => Vector) with Serializable

trait Regularizer extends Serializable {
  def primal(w: Vector): Double
  def dualGradient(w: Vector): Vector
  def lambda: Double

  def dual(v: Vector) = {
    val w = dualGradient(v)
    dot(w,v) - primal(w)
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

class L2Regularizer(val lambda: Double) extends Regularizer {

  def primal(w: Vector) = l2norm(w) / 2
  def dualGradient(w: Vector) = w

  def dual2(v: Vector) = l2norm(v)/2
}

class L1Regularizer(sigma: Double, epsilon: Double) extends Regularizer {

  val B = 1.0 / sigma
  val lambda = epsilon / (B * B)
  val p = sigma / lambda
  println(s"$lambda 0.5|w|_2^2 + $sigma |w|_1")
  def primal(w: Vector) = 1.0/2*l2norm(w) + p*l1norm(w)

  def dualGradient(v: Vector) = {

    val vcopy = v.copy

    val values = vcopy match {
      case (vcopy: SparseVector) => vcopy.values
      case (vcopy: DenseVector) => vcopy.values
    }

    values.transform(v => if (math.abs(v) > p) v - p*math.signum(v) else 0.0)

    vcopy
  }

  def dual2(v: Vector) = v.asInstanceOf[DenseVector].values.map(
    v => math.pow(if(math.abs(v) > p) v - p*math.signum(v) else 0.0,2)
  ).sum / 2
}