#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;
  previous_timestamp_ = 0;

  //initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  //P_ matrix: Initial State Covariance Matrix
  //Initial State based on initialization error
  //Start with Identity Matrix
  MatrixXd I = MatrixXd::Identity(4, 4);
  MatrixXd P_ = MatrixXd(4, 4);
  P_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;

  //initialize F as Identity Matrix
  //F should be identify matrix, but with dt as 1 (in terms of seconds)
  //F*x predicts where object will be after time dt
  int dt = 1;
  MatrixXd F_ = MatrixXd(4, 4);
  F_ << 1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1;

  //Lidar data is only a function of p(px,py), not v(vx,vy)
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  //initialize state vector as lidar
  VectorXd x_ = VectorXd(4);
  x_ << 1, 1, 0, 0;

  //as creation: instantiate Kalman Filter, Tools
  //initialize variables in Kalman Filter as function of Lidar
  MatrixXd Q_ = MatrixXd(4, 4);
  ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);

  noise_ax = 9;
  noise_ay = 9;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    //first measurement
    cout << "EKF: " << endl;
    double px = 0, py = 0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      //Convert radar from polar to cartesian coordinates and initialize state
      double rho = measurement_pack.raw_measurements_[0];
      double phi = measurement_pack.raw_measurements_[1];
      px = rho * cos(phi);
      py = rho * sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      //Initialize state vector (px, py)
      px = measurement_pack.raw_measurements_[0];
      py = measurement_pack.raw_measurements_[1];
    }

    ekf_.x_ << px, py, 0, 0;
    previous_timestamp_ = measurement_pack.timestamp_;
    //done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  //Based on acceleration: update F and Q matrices
  //compute the time elapsed (sec) between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  //Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;
  //calculate Q Matrix: Process Covariance matrix
  //using fixed noise components (noise_ax=9, noise_ay=9)
  float dt_2 = dt   * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
              0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
              dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
              0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    //Radar updates: note we are updating the variables in the ekf class directly
    //Linear Approximation Measurement Conversion via Jocobian as function of x
    //H(Measurement Matrix) => Hj(Jacobian Matrix)
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    //Laser updates: note we are updating the variables in the ekf class directly
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  //print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
