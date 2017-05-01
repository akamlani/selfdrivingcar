#include "kalman_filter.h"
#include <stdlib.h>
#include <math.h>
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  /*
   * Initialize Kalman Filter
   * x = state vector
   * p = Linear Motion: p' = p + vel*delta_t (vel' = vel if no movement)
   */
  x_ = x_in;  //let x be of form x = [px, py, vx, vy] (columnar form: 4x1)
  P_ = P_in;  //State Covariance Matrix
  F_ = F_in;  //State Transition Matrix
  H_ = H_in;  //Measurement Matrix
  R_ = R_in;  //Measurement Noise (variance of RSSI Signal)
  Q_ = Q_in;  //Process noise caused by system (low value of noise: 0.008, most caused by measurement noise)

  long x_size = x_.size();
  I = MatrixXd::Identity(x_size, x_size);
}

void KalmanFilter::Predict() {
  /*
   * Expectation of state without using measurements
   * Let v = Process Noise: v ~ N(0,Q)
   */
  MatrixXd Ft_ = F_.transpose();
  x_ = F_ * x_;                         //x' = F*x + B*u + v (we assume B*u=0: no motion)
  P_ = (F_ * P_ * Ft_) + Q_;            //P' = F*P*F.T + Q
}

void KalmanFilter::Update(const VectorXd &z) {
  /*
   * Perform linear measurement update
   * Let z = H*x' + w
   * Let z = (1 0)*(p',v') (columnar format) (for no velocity)
   * Let w = Measurement Noise: w ~ N(0, R): Gaussian: mean=0, covariance matrix=R
   * Let K = Kalman Gain: Weighting betweeen certainty of estimate and mesurement (influence by noise R)
   */
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;               //PHt = P'*H.T
  MatrixXd S = H_ * PHt + R_;           //S = H*P'*H.T + R
  MatrixXd K = PHt * S.inverse();       //K = P'*H.T*S.inverse

  VectorXd zhat = H_ * x_;              //measurement prediction
  VectorXd y = z - zhat;                //y = z - H*x'

  //Project new estimate
  x_ = x_ + (K * y);                    //x = x' + K*y
  P_ = (I - K*H_) * P_;                 //P = (I - K*H)*P'
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * Let z = h(x') + omega(noise);   omega ~ N(0, R)
    * Let z be function of [rho, phi, rhodot] (columnar format)
    * Normalized Phi between [-Pi, Pi]: add or subtract 2*PI
  */

  double px = x_[0];
  double py = x_[1];
  double vx = x_[2];
  double vy = x_[3];

  double rho = sqrt(px*px + py*py);
  double phi = atan2(py, px);           //atan2 normalizes between [-Pi, Pi]
  double rho_dot = (px*vx + py*vy)/rho;

  MatrixXd Ht = H_.transpose();         //Ht = H.T
  MatrixXd PHt = P_ * Ht;               //PHt = P'*H.T
  MatrixXd S = H_ * PHt + R_;           //S = H*P'*H.T + R
  MatrixXd K = PHt * S.inverse();       //K = P'*H.T*S.inverse

  VectorXd zhat(3);
  zhat << rho, phi, rho_dot;            //measurement prediction
  VectorXd y = z - zhat;                //y = z - h(x')

  //constraint to [-PI, PI]
  double y_phi = y(1);
  if (y_phi > M_PI) { y(1) -= 2*M_PI; }
  else if (y_phi < -M_PI) { y(1) += 2*M_PI; }

  //Project new estimate
  x_ = x_ + (K * y);                    //x = x' + K*y
  P_ = (I - K*H_) * P_;                 //P = (I - K*H)*P'
}
