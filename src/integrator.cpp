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
    

Integrator::Integrator(WavePacket& wp, double atol, double rtol):
_wp{&wp},_atol{atol},_rtol{rtol}{}

State Integrator::integrate(State s0, std::pair<double, double> t_integr) const
{

  auto controlled_stepper=odeint::make_controlled(_atol, _rtol, error_stepper_type());

  auto const [t_0, t_end] = t_integr;
  double dt_init = 0.01;

  integrate_adaptive(
  controlled_stepper,[this](const State& s, State &dsdt, double t){dsdt = _wp->system(s,t);},
  s0, t_0, t_end, dt_init
  );
  return s0;
}

}

