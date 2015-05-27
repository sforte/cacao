import optimizers.coordinate.{BrentMethodOptimizerWithFirstDerivative, BrentMethodOptimizer}

package object optimizers {
  implicit val brent = new BrentMethodOptimizer
//  implicit val brentFirstDer = new BrentMethodOptimizerWithFirstDerivative
}
