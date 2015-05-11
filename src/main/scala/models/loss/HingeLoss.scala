package models.loss

import math.max
import Double.{NegativeInfinity,PositiveInfinity}
import models._

sealed class HingeLoss extends GenericLoss[RealFunction,DifferentiableRealFunction] (

  (y:Double) => (
    (x:Double) => max(0, 1 - x*y),
    (NegativeInfinity, PositiveInfinity)
  ),

  (y:Double) => (
    (a:Double) => y*a,
    (a:Double) => y,
    if (y == 1) (-1.0,0.0) else (0.0,1.0)
  )
)