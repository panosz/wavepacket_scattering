#ifndef WP_INTEGRATOR_DEFINITIONS
#define WP_INTEGRATOR_DEFINITIONS
#include "type_definitions.hpp"
namespace WP{
class WavePacket;


class Integrator
{
public:
  static constexpr double ATOL_DEFAULT=1e-10;
  static constexpr double RTOL_DEFAULT=1e-10;

  Integrator(WavePacket&,
    double atol=ATOL_DEFAULT, double rtol=RTOL_DEFAULT);
  Integrator(Integrator &&) = default;
  Integrator(const Integrator &) = default;
  ~Integrator() = default;

  State integrate(State s0, std::pair<double, double> t_integr) const;

private:

  WavePacket* _wp;
  double _atol, _rtol;
};
}
#endif // !WP_INTEGRATOR_DEFINITIONS
