package com.linkedin.photon.ml.function


import com.linkedin.photon.ml.data.{SimpleObjectProvider, ObjectProvider}
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}
import com.linkedin.photon.ml.util.Utils

/**
 * Class for the logistic loss function: sum_i (w_i*(y_i*log(1 + exp(-(theta'x_i + o_i))) + (1-y_i)*log(1 + exp(theta'x_i + o_i)))),
 * where \theta is the coefficients of the data features to be estimated, (y_i, x_i, o_i, w_i) are the tuple
 * for label, features, offset, and weight of the i'th labeled data point, respectively.
 * Note that the above equation assumes the label y_i \in {0, 1}. However, the code below would also work when y_i \in {-1, 1}.
 * @author xazhang
 * @author dpeng
 */
class LogisticLossFunction(normalizationContext: ObjectProvider[NormalizationContext] = new SimpleObjectProvider[NormalizationContext](NoNormalization)) extends
  GeneralizedLinearModelLossFunction(PointwiseLogisticLossFunction, normalizationContext)

/**
 * A single logistic loss function
 *
 * l(z, y) = - log [1/(1+exp(-z))]         if this is a positive sample
 *
 *         or - log [1 - 1/(1+exp(-z))]    if this is a negative sample
 *
 */
@SerialVersionUID(1L)
object PointwiseLogisticLossFunction extends PointwiseLossFunction {
  /**
   * The sigmoid function 1/(1+exp(-z))
   *
   * @param z z
   * @return The value
   */
  private def sigmoid(z: Double): Double = 1.0 / (1.0 + math.exp(-z))


  /**
   * l(z, y) = - log [1 / (1 + exp(-z))] = log [1 + exp(-z)]            if this is a positive sample
   *
   *           - log [1 - 1/(1+exp(-z))] = log [1 + exp(z)]             if this is a negative sample
   *
   * dl/dz =  - 1 / (1 + exp(z))         if this is a positive sample
   *
   *          1 / (1 + exp(-z))          if this is a negative sample
   *
   * @param margin The margin, i.e. z in l(z, y)
   * @param label The label, i.e. y in l(z, y)
   * @return The value and the 1st derivative
   */
  override def loss(margin: Double, label: Double): (Double, Double) = {
    if (label > 0) {
      // The following is equivalent to log(1 + exp(-margin)) but more numerically stable.
      (Utils.log1pExp(-margin), - sigmoid(-margin))
    } else {
      (Utils.log1pExp(margin), sigmoid(margin))
    }
  }

  /**
   * d^2^l/dz^2^ = sigmoid(z) * (1 - sigmoid(z))
   *
   * @param margin The margin, i.e. z in l(z, y)
   * @param label The label, i.e. y in l(z, y)
   * @return The value and the 2st derivative with respect to z
   */
  override def d2lossdz2(margin: Double, label: Double): Double = {
    val s = sigmoid(margin)
    s * (1 - s)
  }
}
