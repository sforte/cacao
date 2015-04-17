package models

/*
  A function that maps a label y and a value x to a loss value.
  Used both to represent a primal and a dual loss.
 */
trait RealFunction extends ((Double,Double) => Double) with Serializable {
  def domain(y: Double) : (Double,Double)
}

trait DifferentiableRealFunction extends RealFunction {
  def derivative : RealFunction
}

trait DoublyDifferentiableRealFunction extends DifferentiableRealFunction {
  def derivative : DifferentiableRealFunction
}

/*
  Class representing a classification/regression model as defined in the CoCoA paper.
 */
trait PrimalDualModel {
  def primalLoss: RealFunction
  def dualLoss: RealFunction
  def lambda: Double
  /*
    It returns an initialization value on which the dual loss, parameterized by y, is feasible
   */
  def initAlpha(y: Double): Double
}

/*
Class representing a classification/regression model with a differentiable dual loss.
*/
trait PrimalDualModelWithFirstDerivative extends PrimalDualModel {
  override def dualLoss: DifferentiableRealFunction
}

/*
Class representing a classification/regression model with a doubly differentiable dual loss.
*/
trait PrimalDualModelWithSecondDerivative extends PrimalDualModelWithFirstDerivative {
  override def dualLoss: DoublyDifferentiableRealFunction
}