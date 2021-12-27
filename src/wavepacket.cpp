#include <cmath>
#include <tuple>
#include "wavepacket.hpp"

namespace WP{

WavePacket::WavePacket(double Ai,
                       double sigmai,
                       double ki,
                       double vpi): sigma_sq{sigmai*sigmai},
                                    A{Ai},
                                    sigma{sigmai},
                                    k{ki},
                                    vp{vpi}{};

template<>
inline Vector WavePacket::_exponent(const Vector& z) const
{
  return -z.square()/(2*sigma_sq);
}

State WavePacket::system(const State& s, double t) const
  {

   const auto& z = s[0];
   const auto& p = s[1];

   const auto dzdt = p;
   const auto dpdt = - dz(z, t);

   return State{dzdt, dpdt};

  }


std::string WavePacket::_to_string() const{

  return std::string{"<_wavepacket.Wavepacket('"}
  + "A="    + std::to_string(A)     + ", "
  + "sigma=" + std::to_string(sigma) + ", "
  + "k="     + std::to_string(k)     + ", "
  + "vp="    + std::to_string(vp)    + ")>";
}

}
