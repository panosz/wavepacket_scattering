#ifndef WAVEPACKET_DEFINITIONS
#define WAVEPACKET_DEFINITIONS

#include <string>
#include <Eigen/Core>
#include "type_definitions.hpp"
#include "integrator.hpp"

namespace WP{

class Integrator;
class WavePacket
  {

    double sigma_sq;
    public:
    double A, sigma, k, vp;
    WavePacket(double A, double sigma, double k, double vp);
    WavePacket(WavePacket &&) = default;
    WavePacket(const WavePacket &) = default;
    ~WavePacket() = default;
    template <typename T>
    T operator()(T z, T t) const;
    template <typename T>
    T dz(const T& z, const T& t) const;

    Integrator make_integrator(double atol=Integrator::ATOL_DEFAULT,
                               double rtol=Integrator::RTOL_DEFAULT);

    template<typename T>
    inline T _exponent(const T& z) const;
    template<typename T>
      inline T _envelope(const T& z) const;
    template<typename T>
    inline T _phase(const T z, const T t) const;
    template<typename T>
    inline auto _phase_and_envelope(const T& z, const T& t) const;
    State system(const State& s, double t) const;
    std::string _to_string() const;
  };

// Template method implementations.{{{
    template <typename T>
    T WavePacket::operator()(T z, T t) const{
      using namespace std;
      const auto [phase, envelope] = _phase_and_envelope(z, t);
      return envelope * sin(phase);
    }

    template <typename T>
    T WavePacket::dz(const T& z, const T& t) const{
      using namespace std;

      const auto [phase, envelope] = _phase_and_envelope(z, t);
      const auto dz1 = envelope * k * cos(phase);
      const auto dz2 = -z/sigma_sq * envelope * sin(phase);

      return dz1 + dz2;
    }

    template<typename T>
    inline T WavePacket::_exponent(const T& z) const
    {
      return  -(z*z)/(2*sigma_sq);
    }

    template<typename T>
      inline T WavePacket::_envelope(const T& z) const
    {
      using namespace std;
      return A * exp(_exponent(z));
    }

    template<typename T>
    inline T WavePacket::_phase(const T z, const T t) const{
      return k*(z - vp *t);
    }

    template<typename T>
    inline auto WavePacket::_phase_and_envelope(const T& z, const T& t) const{

      using namespace std;

      return std::make_tuple(_phase(z, t), _envelope(z));
    }/*}}}*/

}
#endif // !WAVEPACKET_DEFINITIONS
