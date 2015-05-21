package test

import org.apache.commons.math3.analysis.UnivariateFunction
import org.apache.commons.math3.analysis.differentiation.{DerivativeStructure, FiniteDifferencesDifferentiator}
import models.loss.LogisticLoss
import models.{DoublyDifferentiableRealFunction, RealFunction, DifferentiableRealFunction, Loss}

/**
* Created by simone on 08/05/15.
*/
object DerivativeTest extends App {

  assert(testForAll(new LogisticLoss().conjugate, Array(-1.0,1.0)))

  def testForAll(loss: Loss[DifferentiableRealFunction,RealFunction], iter: Iterable[Double]) = {
    iter.forall(y => testDerivative(loss.apply(y), 0.01))
  }

  def testDerivative(function: DifferentiableRealFunction, step: Double, tol: Double = 10E-1)
    : Boolean = {

    val isDerivative = Range.Double(function.domain._1 + step, function.domain._2, step)
      .forall(x => {

        println(x, differentiate(function, x), function.derivative(x))
        math.abs(differentiate(function, x) - function.derivative(x)) <= math.abs(tol*function.derivative(x))
    })

    function match {
      case (f: DoublyDifferentiableRealFunction) => isDerivative && testDerivative(f.derivative, step, tol)
      case (_) => isDerivative
    }
  }

  def differentiate(function: RealFunction, x: Double, nPoints: Int = 2, step: Double = 0.005): Double = {
    new FiniteDifferencesDifferentiator(nPoints, step)
      .differentiate(new UnivariateFunction {
      override def value(x: Double) = function(x)
    }).value(new DerivativeStructure(1, 1, 0, x)).getPartialDerivative(1)
  }
}

