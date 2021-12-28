#include <Eigen/Core>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>
#include <boost/numeric/odeint/external/eigen/eigen_algebra.hpp>
#include "wavepacket.hpp"
#include "type_definitions.hpp"
namespace WP{
  namespace odeint = boost::numeric::odeint;


using error_stepper_type = odeint::runge_kutta_cash_karp54<State,double,State,double,odeint::vector_space_algebra > ;



using controlled_stepper_type = odeint::controlled_runge_kutta< error_stepper_type > ;
    

Integrator::Integrator(WavePacket& wp):_wp{&wp}{}

State Integrator::integrate(State s0, double t_integr) const
{

  controlled_stepper_type controlled_stepper;
  double dt_init = 0.01;

  integrate_adaptive( controlled_stepper,[this](const State& s, State &dsdt, double t){dsdt = _wp->system(s,t);}, s0, 0.0, t_integr, dt_init);
  return s0;
}

}

