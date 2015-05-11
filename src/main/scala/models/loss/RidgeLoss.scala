package models.loss

import models._
import Double.{NegativeInfinity,PositiveInfinity}

sealed class RidgeLoss extends GenericLoss[RealFunction,DifferentiableRealFunction] (

  (y:Double) => (
    (x:Double) => (x - y) * (x - y),
    (NegativeInfinity,PositiveInfinity)
  ),
  (y:Double) => (
    (a:Double) => a*a/4 + a*y,
    (a:Double) => a/2 + y,
    (NegativeInfinity,PositiveInfinity)
  )
)