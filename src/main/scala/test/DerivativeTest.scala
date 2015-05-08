package test

import org.apache.commons.math3.analysis.differentiation.FiniteDifferencesDifferentiator
import models.{LogisticLossConjugate, LogisticLoss, DifferentiableLoss}

/**
 * Created by simone on 08/05/15.
 */
object DerivativeTest extends App {

  testDerivative(LogisticLossConjugate)

  def testDerivative(function: DifferentiableLoss) {

    val differentiatior = new FiniteDifferencesDifferentiator(10, 0.25)

    differentiatior.differentiate()

    function.derivative
  }
}

