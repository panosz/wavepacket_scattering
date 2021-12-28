#ifndef WP_INTEGRATOR_DEFINITIONS
#define WP_INTEGRATOR_DEFINITIONS
#include "type_definitions.hpp"
namespace WP{
class WavePacket;
class Integrator
{
public:
  explicit Integrator(WavePacket&);
  Integrator(Integrator &&) = default;
  Integrator(const Integrator &) = default;
  ~Integrator() = default;

  State integrate(State s0, double t_integr) const;

private:

  WavePacket* _wp;
};
}
#endif // !WP_INTEGRATOR_DEFINITIONS
